import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
import spacy

from .vocabulary import Vocabulary

spacy_en = spacy.load('en_core_web_sm')
spacy_de = spacy.load('de_core_news_sm')


def tokenize_en(text):
    return [tok.text.lower() for tok in spacy_en.tokenizer(text)]

def tokenize_de(text):
    return [tok.text.lower() for tok in spacy_de.tokenizer(text)]


def build_vocab(min_freq=2):
    data = load_dataset('bentrevett/multi30k', split='train')

    src_sentences = [tokenize_en(item['en']) for item in data]
    tgt_sentences = [tokenize_de(item['de']) for item in data]

    src_vocab = Vocabulary()
    src_vocab.build(src_sentences, min_freq=min_freq)

    tgt_vocab = Vocabulary()
    tgt_vocab.build(tgt_sentences, min_freq=min_freq)

    return src_vocab, tgt_vocab

class Multi30kDataset(Dataset):
    def __init__(self, split, src_vocab, tgt_vocab):
        data = load_dataset('bentrevett/multi30k', split=split)
        self.samples = [
            (
                torch.tensor(src_vocab.encode(tokenize_en(item['en'])), dtype=torch.long),
                torch.tensor(tgt_vocab.encode(tokenize_de(item['de'])), dtype=torch.long),
            )
            for item in data
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def _collate_fn(batch, pad_idx):
    src_batch, tgt_batch = zip(*batch)
    src_padded = pad_sequence(src_batch, batch_first=True, padding_value=pad_idx)
    tgt_padded = pad_sequence(tgt_batch, batch_first=True, padding_value=pad_idx)
    return src_padded, tgt_padded


def get_dataloaders(src_vocab, tgt_vocab, batch_size=128):
    pad_idx = src_vocab.word2idx[Vocabulary.PAD]
    collate = lambda batch: _collate_fn(batch, pad_idx)

    train_ds = Multi30kDataset('train',      src_vocab, tgt_vocab)
    val_ds   = Multi30kDataset('validation', src_vocab, tgt_vocab)
    test_ds  = Multi30kDataset('test',       src_vocab, tgt_vocab)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  collate_fn=collate)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, collate_fn=collate)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False, collate_fn=collate)

    return train_loader, val_loader, test_loader
