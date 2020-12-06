import os
import json
import torch
import argparse
import random
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline

from model import SentenceVAE
from ptb import DefaultTokenizer
from utils import to_var, idx2word, interpolate

# from latent_optimizer import Semantic_Loss

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


def do_one_attack(vae, victim_sentence, victim_model, args):
    def prob_to_label(score):
        return 1 if score > 0.5 else 0

    start_sentence = victim_sentence
    start_label = prob_to_label(victim_model(start_sentence))
    end_sentence = args.reference_sentence
    end_label = prob_to_label(victim_model(end_sentence))
    assert start_label != end_label
    for i in range(args.iter):
        print(f'-------ITERATION {i}-------')
        print(f'Victim Sentence: {start_sentence}')
        print(f'Reference Sentence: {end_sentence}')
        interpolated_sentences = get_interpolations(vae, start_sentence, end_sentence, args)

        print('-------PREDICTIONS-------')
        interpolated_sentences = [start_sentence] + interpolated_sentences + [end_sentence]
        best_adv_prob = None
        for i, sentence in enumerate(interpolated_sentences):
            prob = victim_model(sentence)
            label = prob_to_label(prob)
            if i+1 < len(interpolated_sentences) and label != start_label:
                if best_adv_prob is None or abs(prob - 0.5) < abs(best_adv_prob - 0.5):
                    best_adv_prob = prob
                    end_sentence = sentence
            print(f'{sentence} pred:{prob}')
        if best_adv_prob is None:
            break

    print('-------Attack Result-------')
    print(f'Victim Sentence: {start_sentence} pred:{victim_model(start_sentence)}')
    print(f'Reference Sentence: {end_sentence} pred:{victim_model(sentence)}')

    return end_sentence


def set_rseed(random_seed):
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True


def main(args):
    set_rseed(args.rseed)

    vae = load_vae_model_from_args(args)
    victim_model = load_huggingface_model_from_args(args)
    do_one_attack(vae, args.victim_sentence, victim_model, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    # parser.add_argument('-n', '--num_samples', type=int, default=10)
    parser.add_argument('-m', '--victim_model', type=str, default='distilbert-base-uncased-finetuned-sst-2-english')
    parser.add_argument('--victim_sentence', type=str)
    parser.add_argument('--reference_sentence', type=str)
    parser.add_argument('-st', '--steps', type=int, default=8)
    parser.add_argument('--iter', type=int, default=3)
    parser.add_argument('--rseed', type=int, default=1007)

    parser.add_argument('-dd', '--data_dir', type=str, default='data')
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
