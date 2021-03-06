import os
import io
import json
import torch
import numpy as np
from collections import defaultdict
from torch.utils.data import Dataset
from nltk.tokenize import TweetTokenizer

from utils import OrderedCounter, to_tensor

def DefaultTokenizer():
    return PTB(
        data_dir='data',
        split='valid',
        create_data=False,
        max_sequence_length=60,
        min_occ=1
    )

class PTB(Dataset):

    def __init__(self, data_dir, split, create_data, **kwargs):

        super().__init__()
        self.data_dir = data_dir
        self.split = split
        self.max_sequence_length = kwargs.get('max_sequence_length', 50)
        self.min_occ = kwargs.get('min_occ', 3)

        self.data_file = 'ptb.'+split+'.json'
        self.vocab_file = 'ptb.vocab.json'

        self.tokenizer = TweetTokenizer(preserve_case=False)

        if create_data:
            print("Creating new %s ptb data."%split.upper())
            self._create_data()

        elif not os.path.exists(os.path.join(self.data_dir, self.data_file)):
            print("%s preprocessed file not found at %s. Creating new."%(split.upper(), os.path.join(self.data_dir, self.data_file)))
            self._create_data()

        else:
            self._load_data()


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        idx = str(idx)

        return {
            'input_sequence': np.asarray(self.data[idx]['input_sequence']),
            'target': np.asarray(self.data[idx]['target']),
            'length': self.data[idx]['length']
        }

    @property
    def vocab_size(self):
        return len(self.w2i)

    @property
    def pad_idx(self):
        return self.w2i['<pad>']

    @property
    def sos_idx(self):
        return self.w2i['<sos>']

    @property
    def eos_idx(self):
        return self.w2i['<eos>']

    @property
    def unk_idx(self):
        return self.w2i['<unk>']

    def get_w2i(self):
        return self.w2i

    def get_i2w(self):
        return self.i2w


    def _load_data(self, vocab=True):

        with open(os.path.join(self.data_dir, self.data_file), 'r') as file:
            self.data = json.load(file)
        if vocab:
            with open(os.path.join(self.data_dir, self.vocab_file), 'r') as file:
                vocab = json.load(file)
            self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _load_vocab(self):
        with open(os.path.join(self.data_dir, self.vocab_file), 'r') as vocab_file:
            vocab = json.load(vocab_file)

        self.w2i, self.i2w = vocab['w2i'], vocab['i2w']

    def _encode(self, sentence):
        words = self.tokenizer.tokenize(sentence)

        input = ['<sos>'] + words
        input = input[:self.max_sequence_length]

        target = words[:self.max_sequence_length-1]
        target = target + ['<eos>']

        assert len(input) == len(target), "%i, %i"%(len(input), len(target))
        length = len(input)

        input.extend(['<pad>'] * (self.max_sequence_length-length))
        target.extend(['<pad>'] * (self.max_sequence_length-length))

        input = [self.w2i.get(w, self.w2i['<unk>']) for w in input]
        target = [self.w2i.get(w, self.w2i['<unk>']) for w in target]

        return {
            'input_sequence': input,
            'target': target,
            'length': length,
        }

    def encode(self, sentence):
        encoded = self._encode(sentence)
        for k, v in encoded.items():
            encoded[k] = np.asarray([v])
        return to_tensor(encoded)

    def _load_raw_data(self):
        lines = []

        # Load original ptb data.
        # raw_data_path = os.path.join(self.data_dir, 'ptb.'+self.split+'.txt')
        # with open(raw_data_path, 'r') as file:
        #     lines.extend(file.readlines())

        sst2_split_map = {
            'train': 'train',
            'valid': 'validation',
            'test': 'test',
        }
        # Load SST-2 dataset.
        import nlp
        sst2_data = nlp.load_dataset('glue', 'sst2')[sst2_split_map[self.split]]
        lines.extend(sst2_data['sentence'])

        return lines

    def _create_data(self):

        if self.split == 'train':
            self._create_vocab()
        else:
            self._load_vocab()

        data = defaultdict(dict)
        raw_data = self._load_raw_data()

        for i, line in enumerate(raw_data):
            id = len(data)
            data[id] = self._encode(line)

        with io.open(os.path.join(self.data_dir, self.data_file), 'wb') as data_file:
            data = json.dumps(data, ensure_ascii=False)
            data_file.write(data.encode('utf8', 'replace'))

        self._load_data(vocab=False)

    def _create_vocab(self):

        assert self.split == 'train', "Vocablurary can only be created for training file."

        w2c = OrderedCounter()
        w2i = dict()
        i2w = dict()

        special_tokens = ['<pad>', '<unk>', '<sos>', '<eos>']
        for st in special_tokens:
            i2w[len(w2i)] = st
            w2i[st] = len(w2i)

        raw_data = self._load_raw_data()

        for i, line in enumerate(raw_data):
            words = self.tokenizer.tokenize(line)
            w2c.update(words)

        for w, c in w2c.items():
            if c > self.min_occ and w not in special_tokens:
                i2w[len(w2i)] = w
                w2i[w] = len(w2i)

        assert len(w2i) == len(i2w)

        print("Vocablurary of %i keys created." %len(w2i))

        vocab = dict(w2i=w2i, i2w=i2w)
        with io.open(os.path.join(self.data_dir, self.vocab_file), 'wb') as vocab_file:
            data = json.dumps(vocab, ensure_ascii=False)
            vocab_file.write(data.encode('utf8', 'replace'))

        self._load_vocab()
