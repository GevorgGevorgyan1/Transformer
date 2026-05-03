from collections import Counter


class Vocabulary:
    PAD = '<pad>'
    SOS = '<sos>'
    EOS = '<eos>'
    UNK = '<unk>'

    def __init__(self):
        self.word2idx = {self.PAD: 0, self.SOS: 1, self.EOS: 2, self.UNK: 3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)

    def build(self, tokenized_sentences, min_freq=2):
        counter = Counter(tok for sent in tokenized_sentences for tok in sent)
        for word, freq in sorted(counter.items()):
            if freq >= min_freq and word not in self.word2idx:
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word

    def encode(self, tokens):
        unk = self.word2idx[self.UNK]
        return (
            [self.word2idx[self.SOS]]
            + [self.word2idx.get(t, unk) for t in tokens]
            + [self.word2idx[self.EOS]]
        )

    def decode(self, ids, strip_special=True):
        special = {self.PAD, self.SOS, self.EOS}
        tokens = [self.idx2word.get(i, self.UNK) for i in ids]
        if strip_special:
            tokens = [t for t in tokens if t not in special]
        return ' '.join(tokens)
