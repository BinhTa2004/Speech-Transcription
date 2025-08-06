class SimpleTokenizer:
    def __init__(self, texts, min_freq=1):
        from collections import Counter

        self.special_tokens = ['<pad>', '<bos>', '<eos>', '<unk>']
        counter = Counter()

        for text in texts:
            words = text.lower().strip().split()
            counter.update(words)

        self.word2idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        for word, freq in counter.items():
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)

        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        self.pad_token_id = self.word2idx['<pad>']

    def encode(self, text, max_len=30):
        tokens = [self.word2idx.get(word, self.word2idx['<unk>']) for word in text.lower().strip().split()]
        tokens = [self.word2idx['<bos>']] + tokens + [self.word2idx['<eos>']]
        if len(tokens) < max_len:
            tokens += [self.word2idx['<pad>']] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return tokens

    def decode(self, tokens):
        words = [self.idx2word.get(idx, '<unk>') for idx in tokens]
        return ' '.join(words)
