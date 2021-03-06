import os
import json
import torch
import argparse
import random
import numpy as np
import json
import time

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

from model import SentenceVAE
from ptb import DefaultTokenizer
from utils import to_var, idx2word, interpolate

from lm_scorer.models.auto import AutoLMScorer as LMScorer
from universal_sentence_encoder import UniversalSentenceEncoder

device = torch.device("cuda:0")

def load_vae_model_from_args(args):
    with open(args.data_dir+'/ptb.vocab.json', 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']

    model = SentenceVAE(
        vocab_size=len(w2i),
        sos_idx=w2i['<sos>'],
        eos_idx=w2i['<eos>'],
        pad_idx=w2i['<pad>'],
        unk_idx=w2i['<unk>'],
        max_sequence_length=args.max_sequence_length,
        embedding_size=args.embedding_size,
        rnn_type=args.rnn_type,
        hidden_size=args.hidden_size,
        word_dropout=args.word_dropout,
        embedding_dropout=args.embedding_dropout,
        latent_size=args.latent_size,
        num_layers=args.num_layers,
        bidirectional=args.bidirectional
        )
    tokenizer = DefaultTokenizer()

    if not os.path.exists(args.load_checkpoint):
        raise FileNotFoundError(args.load_checkpoint)

    model.load_state_dict(torch.load(args.load_checkpoint))
    print("Model loaded from %s" % args.load_checkpoint)

    if torch.cuda.is_available():
        model = model.cuda()

    model.eval()
    return {
      'model': model,
      'tokenizer': tokenizer,
      'w2i': w2i,
      'i2w': i2w,
    }



def load_huggingface_model_from_args(args):
    victim_model = args.victim_model
    tokenizer = AutoTokenizer.from_pretrained(victim_model)
    model = AutoModelForSequenceClassification.from_pretrained(victim_model).to(torch.device("cuda"))
    def pipeline(sentence):
      encode = tokenizer(sentence, return_tensors="pt").to(torch.device("cuda"))
      logits = model(**encode)[0]
      return torch.softmax(logits, dim=1).tolist()[0][1]
    return pipeline


def get_interpolations(vae, sample_start, sample_end, args):
    model = vae['model']
    tokenizer = vae['tokenizer']
    w2i = vae['w2i']
    i2w = vae['i2w']
    # Initialize semantic loss
    # sl = Semantic_Loss()

    start_encode = tokenizer.encode(sample_start)
    end_encode = tokenizer.encode(sample_end)
    with torch.no_grad():
        z1 = model._encode(**start_encode)
        z1_hidden = z1['z'].cpu()[0]

        z2 = model._encode(**end_encode)
        z2_hidden = z2['z'].cpu()[0]

    z_hidden = to_var(torch.from_numpy(interpolate(start=z1_hidden, end=z2_hidden, steps=args.steps)).float())

    if args.rnn_type == "lstm":
        z1_cell_state = z1['z_cell_state'].cpu()[0].squeeze()
        z2_cell_state = z2['z_cell_state'].cpu()[0].squeeze()

        # print(z1_cell_state.shape)

        z_cell_states = \
            to_var(torch.from_numpy(interpolate(start=z1_cell_state, end=z2_cell_state, steps=args.steps)).float())

        samples, _ = model.inference(z=z_hidden, z_cell_state=z_cell_states)
    else:
        samples, _ = model.inference(z=z_hidden, z_cell_state=None)
    # print('-------INTERPOLATION-------')

    interpolated_sentences = idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'])
    # For each sentence, get the perplexity and show it
    # for sentence in interpolated_sentences:
        # print(sentence + "\t\t" + str(sl.get_perplexity(sentence)))
        # print(sentence)

    return interpolated_sentences


def prob_to_label(score):
        return 1 if score > 0.5 else 0

def do_one_attack(vae, victim_sentence, reference_sentence, victim_model, ppl, use, args):
    start_sentence = victim_sentence
    start_prob = victim_model(start_sentence)
    start_label = prob_to_label(start_prob)
    end_sentence = reference_sentence
    end_prob = victim_model(end_sentence)
    end_label = prob_to_label(end_prob)
    assert start_label != end_label
    best_use = use(start_sentence, end_sentence)
    is_initial = True

    print(f'\n-------Initial Inputs-------')
    print(f'Victim Sentence: {start_sentence} pred:{start_prob} PPL:{ppl(start_sentence):.0f} USE:{use(start_sentence, start_sentence):.2f}')
    print(f'Reference Sentence: {end_sentence} pred:{end_prob} PPL:{ppl(end_sentence):.0f} USE:{use(start_sentence, end_sentence):.2f}')

    for i in range(args.iter):
        print(f'\n-------ITERATION {i}-------')
        print(f'Best Adv Sentence: {end_sentence} pred:{end_prob}')
        interpolated_sentences = get_interpolations(vae, start_sentence, end_sentence, args)

        if args.verbose:
            print('-------PREDICTIONS-------')
            print(f'Pred & Sentence & PPL & USE \\\\')
        interpolated_sentences = [start_sentence] + interpolated_sentences + [end_sentence]
        interpolated_sentences = [s.replace("<eos>", "") for s in interpolated_sentences]
        found_next = False
        for i, sentence in enumerate(interpolated_sentences):
            if i > 0 and sentence == interpolated_sentences[i-1]:
                continue
            prob = victim_model(sentence)
            label = prob_to_label(prob)
            if i+1 < len(interpolated_sentences) and label != start_label:
                if args.most_similar and not found_next:
                    end_sentence = sentence
                    found_next = True

                if not args.most_similar:
                    new_use = use(start_sentence, sentence)
                    if new_use > best_use:
                        is_initial = False
                        best_use = new_use
                        end_sentence = sentence

            if args.verbose:
                print(f'{prob:.3f} & {sentence} & {ppl(sentence):.0f} & {use(start_sentence, sentence):.2f} \\\\')

    print('-------Attack Result-------')
    print(f'Victim Sentence: {start_sentence} pred:{victim_model(start_sentence)} PPL:{ppl(start_sentence):.0f} USE:{use(start_sentence, start_sentence):.2f}')
    print(f'Best Adv Sentence: {end_sentence} pred:{victim_model(end_sentence)} PPL:{ppl(end_sentence):.0f} USE:{use(start_sentence, end_sentence):.2f}')

    return end_sentence


def do_n_attacks(vae, victim_sentence, victim_model, ppl, use, args):
    victim_label = prob_to_label(victim_model(victim_sentence))

    # Get sentences with the different label.
    import nlp
    sst2_data = nlp.load_dataset('glue', 'sst2')['train']
    sentences = [e['sentence'] for e in sst2_data if e['label'] != victim_label and prob_to_label(victim_model(e['sentence'])) != victim_label]
    sentences = random.sample(sentences, args.n_attacks)

    print(f'\n-------Running {args.n_attacks} Attacks-------')
    print(f'sentences = {sentences}')

    use_adv = []
    for s in sentences:
        adv = do_one_attack(vae, victim_sentence, s, victim_model, ppl, use, args)
        use_adv.append((use(victim_sentence, adv), adv))

    use_adv = sorted(use_adv, reverse=True)
    best_adv = use_adv[0][1]
    print(f'\n-------Result from {args.n_attacks} Attacks-------')
    print(f'use_adv = {use_adv}')
    print(f'Victim Sentence: {victim_sentence} pred:{victim_model(victim_sentence)} PPL:{ppl(victim_sentence):.0f} USE:{use(victim_sentence, victim_sentence):.2f}')
    print(f'Best Adv Sentence: {best_adv} pred:{victim_model(best_adv)} PPL:{ppl(best_adv):.0f} USE:{use(victim_sentence, best_adv):.2f}')
    return best_adv


def do_evaluation(vae, victim_model, ppl, use, args):
    # Get validations examples.
    import nlp
    sst2_data = nlp.load_dataset('glue', 'sst2')['validation']
    victim_sentences = sst2_data['sentence']
    victim_sentences = random.sample(victim_sentences, args.n_eval)

    advs = []
    for i, vs in enumerate(victim_sentences):
        print(f'\n-------Start Evaluation {i}/{args.n_eval}-------')
        adv = do_n_attacks(vae, vs, victim_model, ppl, use, args)
        print(f'\n-------Finished Evaluation {i}/{args.n_eval}-------')
        advs.append(adv)

    print(f'\n-------Evaluation Results-------')
    summary = []
    for vic, adv in zip(victim_sentences, advs):
        summary.append({
            'victim_sentence': vic,
            'victim_pred': victim_model(vic),
            'victim_ppl': ppl(vic),
            'adv_sentence': adv,
            'adv_pred': victim_model(adv),
            'adv_ppl': ppl(adv),
            'use': use(vic, adv),
        })
    print(summary)

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime()) + '.json'
    save_eval_path = os.path.join(args.save_eval_path, ts)
    print(f'Saving summary to {save_eval_path}...')
    with open(save_eval_path, 'w') as outfile:
        json.dump(summary, outfile)

def set_rseed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    set_rseed(args.rseed)

    if args.ppl_use:
        print(f'Loading GPT-2...')
        lm_scorer = LMScorer.from_pretrained("gpt2", device="cuda:0", batch_size=1)
        print(f'Loading UniversalSentenceEncoder...')
        u = UniversalSentenceEncoder()
        def ppl(s):
            return -lm_scorer.sentence_score(s, log=True)
        def use(s1, s2):
            return u.cos_sim(s1, s2)
    else:
        def ppl(s):
            return 0
        def use(s1, s2):
            return 0

    print(f'Loading the VAE model...')
    vae = load_vae_model_from_args(args)
    print(f'Loading the victim model from HuggingFace...')
    victim_model = load_huggingface_model_from_args(args)

    if args.n_eval > 0:
        print(f'\n-------Evaluation Mode-------')
        do_evaluation(vae, victim_model, ppl, use, args)
    else:
        print(f'\n-------Debug Mode-------')
        if args.reference_sentence is None:
            do_n_attacks(vae, args.victim_sentence, victim_model, ppl, use, args)
        else:
            do_one_attack(vae, args.victim_sentence, args.reference_sentence, victim_model, ppl, use, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-m', '--victim_model', type=str, default='distilbert-base-uncased-finetuned-sst-2-english')
    parser.add_argument('--victim_sentence', type=str, default=None)
    parser.add_argument('--reference_sentence', type=str, default=None)
    parser.add_argument('--n_attacks', type=int, default=1)
    parser.add_argument('-st', '--steps', type=int, default=8)
    parser.add_argument('--iter', type=int, default=3)
    parser.add_argument('--rseed', type=int, default=1007)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--most_similar', action='store_true')
    parser.add_argument('--ppl_use', action='store_true')
    parser.add_argument('--n_eval', type=int, default=0)

    parser.add_argument('-dd', '--data_dir', type=str, default='data')
    parser.add_argument('--save_eval_path', type=str, default='eval')
    parser.add_argument('-ms', '--max_sequence_length', type=int, default=50)
    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert 0 <= args.word_dropout <= 1

    main(args)
