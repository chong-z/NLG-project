import os
import json
import time
import torch
import argparse
import numpy as np
from multiprocessing import cpu_count
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from collections import OrderedDict, defaultdict

from ptb import PTB
from utils import to_var, idx2word, expierment_name, interpolate
from model import SentenceVAE

from latent_optimizer import Semantic_Loss

import itertools
import random


def main(args):

    # Load the vocab
    with open(args.data_dir+'/ptb.vocab.json', 'r') as file:
        vocab = json.load(file)

    w2i, i2w = vocab['w2i'], vocab['i2w']

    ts = time.strftime('%Y-%b-%d-%H:%M:%S', time.gmtime())

    splits = ['train', 'valid'] + (['test'] if args.test else [])

    # Initialize semantic loss
    sl = Semantic_Loss()

    datasets = OrderedDict()
    for split in splits:
        datasets[split] = PTB(
            data_dir=args.data_dir,
            split=split,
            create_data=args.create_data,
            max_sequence_length=args.max_sequence_length,
            min_occ=args.min_occ
        )

    params = dict(
        vocab_size=datasets['train'].vocab_size,
        sos_idx=datasets['train'].sos_idx,
        eos_idx=datasets['train'].eos_idx,
        pad_idx=datasets['train'].pad_idx,
        unk_idx=datasets['train'].unk_idx,
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
    model = SentenceVAE(**params)

    if torch.cuda.is_available():
        model = model.cuda()

    print(model)

    if args.tensorboard_logging:
        writer = SummaryWriter(os.path.join(args.logdir, expierment_name(args, ts)))
        writer.add_text("model", str(model))
        writer.add_text("args", str(args))
        writer.add_text("ts", ts)

    save_model_path = os.path.join(args.save_model_path, ts)
    os.makedirs(save_model_path)

    with open(os.path.join(save_model_path, 'model_params.json'), 'w') as f:
        json.dump(params, f, indent=4)

    def kl_anneal_function(anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1/(1+np.exp(-k*(step-x0))))
        elif anneal_function == 'linear':
            return min(1, step/x0)

    def perplexity_anneal_function(anneal_function, step, k, x0):
        if anneal_function == 'logistic':
            return float(1/ 1+np.exp(-k*(step-x0)))
        elif anneal_function == 'linear':
            return min(1, (step/x0))

    NLL = torch.nn.NLLLoss(ignore_index=datasets['train'].pad_idx, reduction='sum')
    def loss_fn(logp, target, length, mean, logv, anneal_function, step, k, x0, \
        batch_perplexity, perplexity_anneal_function):

        # cut-off unnecessary padding from target, and flatten
        target = target[:, :torch.max(length).item()].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))

        # Negative Log Likelihood
        NLL_loss = NLL(logp, target)

        # KL Divergence
        KL_loss = -0.5 * torch.sum(1 + logv - mean.pow(2) - logv.exp())
        KL_weight = kl_anneal_function(anneal_function, step, k, x0)

        # Perplexity
        perp_loss = batch_perplexity
        perp_weight = perplexity_anneal_function(anneal_function, step, k, x0)

        return NLL_loss, KL_loss, KL_weight, perp_loss, perp_weight


    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    step = 0
    for epoch in range(args.epochs):

        # Keep track of epoch loss
        epoch_loss = []

        for split in splits:

            data_loader = DataLoader(
                dataset=datasets[split],
                batch_size=args.batch_size,
                shuffle=split=='train',
                num_workers=cpu_count(),
                pin_memory=torch.cuda.is_available()
            )

            tracker = defaultdict(tensor)

            # Enable/Disable Dropout
            if split == 'train':
                model.train()
            else:
                model.eval()

            batch_t_start = None

            for iteration, batch in enumerate(data_loader):

                if batch_t_start:
                    batch_run_time = time.time() - batch_t_start
                    # print("Batch run time: " + str(batch_run_time))
                batch_t_start = time.time()


                batch_size = batch['input_sequence'].size(0)

                for k, v in batch.items():
                    if torch.is_tensor(v):
                        batch[k] = to_var(v)

                # Get the original sentences in this batch
                batch_sentences = idx2word(batch['input_sequence'], i2w=i2w, pad_idx=w2i['<pad>'])
                # Remove the first tag
                batch_sentences = [x.replace("<sos>", "") for x in batch_sentences]

                # Forward pass
                (logp, mean, logv, z), states = model(**batch)


                # Choose some random pairs of samples within the batch
                #  to get latent representations for
                batch_index_pairs = list(itertools.combinations(np.arange(batch_size), 2))
                random.shuffle(batch_index_pairs)
                batch_index_pairs = batch_index_pairs[:args.perplexity_samples_per_batch]

                batch_perplexity = []

                # If we start the perplexity
                start_perplexity = epoch > 10

                # If we should have perplexity loss
                if start_perplexity and args.perplexity_loss:
                    # For each pair, get the intermediate representations in the latent space
                    for index_pair in batch_index_pairs:

                        with torch.no_grad():
                            z1_hidden = states['z'][index_pair[0]].cpu()
                            z2_hidden = states['z'][index_pair[1]].cpu()

                        z_hidden = to_var(torch.from_numpy(interpolate(start=z1_hidden, end=z2_hidden, steps=1)).float())

                        if args.rnn_type == "lstm":

                            with torch.no_grad():
                                z1_cell_state = states['z_cell_state'].cpu().squeeze()[index_pair[0]]
                                z2_cell_state = states['z_cell_state'].cpu().squeeze()[index_pair[1]]

                            z_cell_states = \
                                to_var(torch.from_numpy(interpolate(start=z1_cell_state, end=z2_cell_state, steps=1)).float())

                            samples, _ = model.inference(z=z_hidden, z_cell_state=z_cell_states)
                        else:
                            samples, _ = model.inference(z=z_hidden, z_cell_state=None)

                        # Check interpolated sentences
                        interpolated_sentences = idx2word(samples, i2w=i2w, pad_idx=w2i['<pad>'])
                        # For each sentence, get the perplexity and show it
                        perplexities = []
                        for sentence in interpolated_sentences:
                            perplexities.append(sl.get_perplexity(sentence))
                        avg_sample_perplexity = sum(perplexities) / len(perplexities)
                        batch_perplexity.append(avg_sample_perplexity)
                    # Calculate batch perplexity
                    avg_batch_perplexity = sum(batch_perplexity) / len(batch_perplexity)

                    # loss calculation
                    NLL_loss, KL_loss, KL_weight, perp_loss, perp_weight = loss_fn(logp, batch['target'],
                        batch['length'], mean, logv, args.anneal_function, step, \
                            args.k, args.x0, avg_batch_perplexity, perplexity_anneal_function)

                    loss = ((NLL_loss + KL_weight * KL_loss) / batch_size) + (perp_loss * perp_weight)

                else: # Epochs < X, so train without perplexity
                    # loss calculation
                    NLL_loss, KL_loss, KL_weight, perp_loss, perp_weight = loss_fn(logp, batch['target'],
                        batch['length'], mean, logv, args.anneal_function, step, \
                            args.k, args.x0, 0, perplexity_anneal_function)

                    loss = (NLL_loss + KL_weight * KL_loss) / batch_size


                # Turn model back into train, since inference changed to eval
                if split == 'train':
                    model.train()
                else:
                    model.eval()

                # backward + optimization
                if split == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    step += 1

                    # Add loss
                    epoch_loss.append(loss.item())

                # bookkeepeing
                tracker['ELBO'] = torch.cat((tracker['ELBO'], loss.data.view(1, -1)), dim=0)

                if args.tensorboard_logging:
                    writer.add_scalar("%s/ELBO" % split.upper(), loss.item(), epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/NLL Loss" % split.upper(), NLL_loss.item() / batch_size,
                                      epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Loss" % split.upper(), KL_loss.item() / batch_size,
                                      epoch*len(data_loader) + iteration)
                    writer.add_scalar("%s/KL Weight" % split.upper(), KL_weight,
                                      epoch*len(data_loader) + iteration)

                if iteration % args.print_every == 0 or iteration+1 == len(data_loader):
                    print("%s Batch %04d/%i, Loss %9.4f, NLL-Loss %9.4f, KL-Loss %9.4f, KL-Weight %6.3f, Perp-loss %9.4f, Perp-weight %6.3f"
                          % (split.upper(), iteration, len(data_loader)-1, loss.item(), NLL_loss.item()/batch_size,
                          KL_loss.item()/batch_size, KL_weight, perp_loss, perp_weight))

                if split == 'valid':
                    if 'target_sents' not in tracker:
                        tracker['target_sents'] = list()
                    tracker['target_sents'] += idx2word(batch['target'].data, i2w=datasets['train'].get_i2w(),
                                                        pad_idx=datasets['train'].pad_idx)
                    tracker['z'] = torch.cat((tracker['z'], z.data), dim=0)

            print("%s Epoch %02d/%i, Mean ELBO %9.4f" % (split.upper(), epoch, args.epochs, tracker['ELBO'].mean()))

            if args.tensorboard_logging:
                writer.add_scalar("%s-Epoch/ELBO" % split.upper(), torch.mean(tracker['ELBO']), epoch)

            # save a dump of all sentences and the encoded latent space
            if split == 'valid':
                dump = {'target_sents': tracker['target_sents'], 'z': tracker['z'].tolist()}
                if not os.path.exists(os.path.join('dumps', ts)):
                    os.makedirs('dumps/'+ts)
                with open(os.path.join('dumps/'+ts+'/valid_E%i.json' % epoch), 'w') as dump_file:
                    json.dump(dump,dump_file)

            # save checkpoint
            if split == 'train':
                checkpoint_path = os.path.join(save_model_path, "E%i.pytorch" % epoch)
                torch.save(model.state_dict(), checkpoint_path)
                print("Model saved at %s" % checkpoint_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--create_data', action='store_true')
    parser.add_argument('--max_sequence_length', type=int, default=60)
    parser.add_argument('--min_occ', type=int, default=1)
    parser.add_argument('--test', action='store_true')

    parser.add_argument('-ep', '--epochs', type=int, default=10)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.001)

    parser.add_argument('-eb', '--embedding_size', type=int, default=300)
    parser.add_argument('-rnn', '--rnn_type', type=str, default='gru')
    parser.add_argument('-hs', '--hidden_size', type=int, default=256)
    parser.add_argument('-nl', '--num_layers', type=int, default=1)
    parser.add_argument('-bi', '--bidirectional', action='store_true')
    parser.add_argument('-ls', '--latent_size', type=int, default=16)
    parser.add_argument('-wd', '--word_dropout', type=float, default=0)
    parser.add_argument('-ed', '--embedding_dropout', type=float, default=0.5)

    parser.add_argument('-af', '--anneal_function', type=str, default='logistic')
    parser.add_argument('-k', '--k', type=float, default=0.0025)
    parser.add_argument('-x0', '--x0', type=int, default=2500)

    parser.add_argument('-v', '--print_every', type=int, default=50)
    parser.add_argument('-tb', '--tensorboard_logging', action='store_true')
    parser.add_argument('-log', '--logdir', type=str, default='logs')
    parser.add_argument('-bin', '--save_model_path', type=str, default='bin')

    parser.add_argument('-psb', '--perplexity_samples_per_batch', type=int, default=10)
    parser.add_argument('--perplexity_loss', action='store_true')

    args = parser.parse_args()

    args.rnn_type = args.rnn_type.lower()
    args.anneal_function = args.anneal_function.lower()

    assert args.rnn_type in ['rnn', 'lstm', 'gru']
    assert args.anneal_function in ['logistic', 'linear']
    assert 0 <= args.word_dropout <= 1

    import time
    start_t = time.time()
    main(args)
    print("Total train time: " + str(time.time() - start_t))
