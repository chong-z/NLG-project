import os
import json
import torch
import argparse
import random
import numpy as np

from semi_attack import *

from lm_scorer.models.auto import AutoLMScorer as LMScorer
from universal_sentence_encoder import UniversalSentenceEncoder


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
    if args.reference_sentence is None:
        do_n_attacks(vae, args.victim_sentence, victim_model, ppl, use, args)
    else:
        do_one_attack(vae, args.victim_sentence, args.reference_sentence, victim_model, ppl, use, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-c', '--load_checkpoint', type=str)
    parser.add_argument('-m', '--victim_model', type=str, default='distilbert-base-uncased-finetuned-sst-2-english')
    parser.add_argument('--victim_sentence', type=str)
    parser.add_argument('--reference_sentence', type=str, default=None)
    parser.add_argument('--n_attacks', type=int, default=1)
    parser.add_argument('-st', '--steps', type=int, default=8)
    parser.add_argument('--iter', type=int, default=3)
    parser.add_argument('--rseed', type=int, default=1007)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('--most_similar', action='store_true')
    parser.add_argument('--ppl_use', action='store_true')

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
