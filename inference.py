import os
import json
import torch
import argparse

from model import SentenceVAE
from ptb import DefaultTokenizer
from utils import to_var, idx2word, interpolate

from latent_optimizer import Semantic_Loss


def main(args):
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

    # Initialize semantic loss
    sl = Semantic_Loss()

    # with torch.no_grad():
    #     samples, z = model.inference(n=args.num_samples)
    print('----------SAMPLES----------')

    sample_start = "a sometimes tedious film ."
    sample_end = "a deep and meaningful film ."

    sample_start = "a sometimes tedious film involving dull characters and boring motives."
    sample_end = "a deep and meaningful film that refreshes the soul."

    print(sample_start)
    print(sample_end)

    # print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')

    # z0 = torch.randn([args.latent_size]).numpy()
    # z2 = torch.randn([args.latent_size]).numpy()
    start_encode = tokenizer.encode(sample_start)
    # print(start_encode)
    # end_encode = tokenizer.encode("it 's a charming and often affecting journey .")
    end_encode = tokenizer.encode(sample_end)
    with torch.no_grad():
        z1 = model._encode(**start_encode)
        z1_hidden = z1['z'].cpu()[0]

        z2 = model._encode(**end_encode)
        z2_hidden = z2['z'].cpu()[0]

    z_hidden = to_var(torch.from_numpy(interpolate(start=z1_hidden, end=z2_hidden, steps=8)).float())

    if args.rnn_type == "lstm":
        z1_cell_state = z1['z_cell_state'].cpu()[0].squeeze()
        z2_cell_state = z2['z_cell_state'].cpu()[0].squeeze()

        # print(z1_cell_state.shape)

        z_cell_states = \
            to_var(torch.from_numpy(interpolate(start=z1_cell_state, end=z2_cell_state, steps=8)).float())

        samples, _ = model.inference(z=z_hidden, z_cell_state=z_cell_states)
    else:
        samples, _ = model.inference(z=z_hidden, z_cell_state=None)
    print('-------INTERPOLATION-------')

    interpolated_sentences = idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'])
    # For each sentence, get the perplexity and show it
    for sentence in interpolated_sentences:
        print(sentence + "\t\t" + str(sl.get_perplexity(sentence)))

    # print(*idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>']), sep='\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-n', '--num_samples', type=int, default=10)

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
