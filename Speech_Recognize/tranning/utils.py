import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torchaudio.transforms as T

class AudioCaptionDataset(Dataset):
    def __init__(self, audio_paths, captions, tokenizer, sr=16000, n_mels=64, max_len=30):
        self.audio_paths = audio_paths
        self.captions = captions
        self.tokenizer = tokenizer
        self.sr = sr
        self.n_mels = n_mels
        self.max_len = max_len
        self.mel_transform = T.MelSpectrogram(sample_rate=sr, n_mels=n_mels)

    def __getitem__(self, idx):
        # Load audio
        waveform, orig_sr = torchaudio.load(self.audio_paths[idx])
        if orig_sr != self.sr:
            waveform = T.Resample(orig_sr, self.sr)(waveform)
            
        max_len_audio = self.sr * 25  # 10 giây
        waveform = waveform[:, :max_len_audio]
        if waveform.shape[1] < max_len_audio:
            pad = max_len_audio - waveform.shape[1]
            waveform = torch.nn.functional.pad(waveform, (0, pad))
        
        mel_spec = self.mel_transform(waveform)
        log_mel = torch.log1p(mel_spec)
        log_mel = (log_mel - log_mel.mean()) / log_mel.std()  # Normalize
        
        log_mel = log_mel.repeat(3, 1, 1) #mel_spec only 1 channel but resnet have 3 channel
        
        # Tokenize caption
        caption_text = self.captions[idx]
        caption_tokens = self.tokenizer.encode(caption_text, max_len=self.max_len)
        caption_tensor = torch.tensor(caption_tokens, dtype=torch.long)

        return log_mel, caption_tensor  # Now return tensor #log_mel.squeeze(0)

    def __len__(self):
        return len(self.audio_paths)


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
