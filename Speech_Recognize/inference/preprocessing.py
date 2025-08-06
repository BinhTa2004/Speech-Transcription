import torch
import librosa
import numpy as np
import torchaudio.transforms as T

def preprocess_audio(filepath, sr=16000, n_mels=64, max_duration=10):
    """
    Load audio file (wav or mp3), resample, convert to log-mel spectrogram, normalize, 
    and prepare for model input.
    
    Args:
        filepath (str): path to audio file
        sr (int): target sample rate
        n_mels (int): number of mel bands
        max_duration (int): max duration (seconds) to pad/truncate audio

    Returns:
        Tensor: processed log-mel spectrogram tensor (shape: [3, n_mels, time_steps])
    """

    try:
        # Use librosa to handle mp3/wav automatically
        y, orig_sr = librosa.load(filepath, sr=sr, mono=True)
    except Exception as e:
        print(f"Error loading audio: {e}")
        raise

    # Ensure length is consistent (pad/truncate to max_duration seconds)
    max_len_samples = sr * max_duration
    if len(y) > max_len_samples:
        y = y[:max_len_samples]
    else:
        pad = max_len_samples - len(y)
        y = np.pad(y, (0, pad), mode='constant')

    # Convert to tensor
    waveform = torch.tensor(y).unsqueeze(0)

    # Mel Spectrogram transform
    mel_transform = T.MelSpectrogram(
        sample_rate=sr,
        n_mels=n_mels,
        n_fft=1024,
        hop_length=512,
    )
    mel_spec = mel_transform(waveform)
    log_mel = torch.log1p(mel_spec)

    # Normalize
    log_mel = (log_mel - log_mel.mean()) / log_mel.std()

    # Repeat to 3 channels for CNN encoder (like ResNet)
    log_mel = log_mel.repeat(3, 1, 1)  # shape: [3, n_mels, time_steps]

    return log_mel

def generate_caption(model, mel, tokenizer, max_len=30, device='cpu'):
    model.eval()
    mel = mel.unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encoder(mel)
        input_word = torch.tensor([[tokenizer.word2idx['<bos>']]]).to(device)
        captions = []
        hidden = None
        for _ in range(max_len):
            embedding = model.decoder.embed(input_word)
            lstm_input = torch.cat((features.unsqueeze(1), embedding), dim=2)
            output, hidden = model.decoder.lstm(lstm_input, hidden)
            output_vocab = model.decoder.fc(output.squeeze(1))
            predicted = output_vocab.argmax(1)
            word = tokenizer.idx2word[predicted.item()]
            if word == '<eos>':
                break
            captions.append(word)
            input_word = predicted.unsqueeze(0)
    return ' '.join(captions)


def generate_caption_deploy(model, mel, tokenizer, max_len=30, device='cpu'):
    model.eval()
    mel = mel.unsqueeze(0).to(device)
    with torch.no_grad():
        features = model.encoder(mel)
        input_word = torch.tensor([[tokenizer["word2idx"]["<bos>"]]]).to(device)
        captions = []
        hidden = None
        for _ in range(max_len):
            embedding = model.decoder.embed(input_word)
            lstm_input = torch.cat((features.unsqueeze(1), embedding), dim=2)
            output, hidden = model.decoder.lstm(lstm_input, hidden)
            output_vocab = model.decoder.fc(output.squeeze(1))
            predicted = output_vocab.argmax(1)
            word = tokenizer["idx2word"][predicted.item()]
            if word == '<eos>':
                break
            captions.append(word)
            input_word = predicted.unsqueeze(0)
    return ' '.join(captions)


import torch
import torch.nn.functional as F

def generate_caption_optimized(model, mel, tokenizer, max_len=30, device='cpu', beam_size=3):
    model.eval()
    mel = mel.unsqueeze(0).to(device)  # [1, T, F]

    with torch.no_grad():
        encoder_out = model.encoder(mel)  # [1, T', D]

        if beam_size == 1:
            # === GREEDY DECODING ===
            input_word = torch.tensor([[tokenizer.word2idx['<bos>']]]).to(device)
            hidden = None
            caption = []

            for _ in range(max_len):
                emb = model.decoder.embed(input_word)
                lstm_input = torch.cat((encoder_out.unsqueeze(1), emb), dim=2)
                output, hidden = model.decoder.lstm(lstm_input, hidden)
                logits = model.decoder.fc(output.squeeze(1))
                next_token = logits.argmax(dim=1).item()
                word = tokenizer.idx2word[next_token]

                if word == '<eos>': break
                caption.append(word)
                input_word = torch.tensor([[next_token]]).to(device)

            return ' '.join(caption)

        else:
            # === BEAM SEARCH ===
            sequences = [[
                [tokenizer.word2idx['<bos>']],  # token sequence
                0.0,                            # cumulative log prob
                None                            # hidden state
            ]]

            for _ in range(max_len):
                all_candidates = []
                for seq, score, hidden in sequences:
                    input_word = torch.tensor([[seq[-1]]]).to(device)
                    emb = model.decoder.embed(input_word)
                    lstm_input = torch.cat((encoder_out.unsqueeze(1), emb), dim=2)
                    output, new_hidden = model.decoder.lstm(lstm_input, hidden)
                    logits = model.decoder.fc(output.squeeze(1))
                    log_probs = F.log_softmax(logits, dim=1)

                    topk_log_probs, topk_ids = log_probs.topk(beam_size)
                    for i in range(beam_size):
                        token = topk_ids[0, i].item()
                        new_seq = seq + [token]
                        new_score = score + topk_log_probs[0, i].item()
                        all_candidates.append([new_seq, new_score, new_hidden])

                # Giữ lại k câu tốt nhất
                sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

                # Nếu tất cả đã kết thúc bằng <eos> thì dừng sớm
                if all(tokenizer.idx2word[seq[-1]] == '<eos>' for seq, _, _ in sequences):
                    break

            # Chọn caption tốt nhất
            best_seq = sequences[0][0][1:]  # Bỏ <bos>
            caption = []
            for idx in best_seq:
                word = tokenizer.idx2word[idx]
                if word == '<eos>': break
                caption.append(word)

            return ' '.join(caption)


def generate_caption_optimized_deploy(model, mel, tokenizer, max_len=30, device='cpu', beam_size=3):
    model.eval()
    mel = mel.unsqueeze(0).to(device)  # [1, T, F]

    with torch.no_grad():
        encoder_out = model.encoder(mel)  # [1, T', D]

        if beam_size == 1:
            # === GREEDY DECODING ===
            input_word = torch.tensor([[tokenizer["word2idx"]['<bos>']]]).to(device)
            hidden = None
            caption = []

            for _ in range(max_len):
                emb = model.decoder.embed(input_word)
                lstm_input = torch.cat((encoder_out.unsqueeze(1), emb), dim=2)
                output, hidden = model.decoder.lstm(lstm_input, hidden)
                logits = model.decoder.fc(output.squeeze(1))
                next_token = logits.argmax(dim=1).item()
                word = tokenizer["idx2word"][next_token]

                if word == '<eos>': break
                caption.append(word)
                input_word = torch.tensor([[next_token]]).to(device)

            return ' '.join(caption)

        else:
            # === BEAM SEARCH ===
            sequences = [[
                [tokenizer["word2idx"]['<bos>']],  # token sequence
                0.0,                            # cumulative log prob
                None                            # hidden state
            ]]

            for _ in range(max_len):
                all_candidates = []
                for seq, score, hidden in sequences:
                    input_word = torch.tensor([[seq[-1]]]).to(device)
                    emb = model.decoder.embed(input_word)
                    lstm_input = torch.cat((encoder_out.unsqueeze(1), emb), dim=2)
                    output, new_hidden = model.decoder.lstm(lstm_input, hidden)
                    logits = model.decoder.fc(output.squeeze(1))
                    log_probs = F.log_softmax(logits, dim=1)

                    topk_log_probs, topk_ids = log_probs.topk(beam_size)
                    for i in range(beam_size):
                        token = topk_ids[0, i].item()
                        new_seq = seq + [token]
                        new_score = score + topk_log_probs[0, i].item()
                        all_candidates.append([new_seq, new_score, new_hidden])

                # Giữ lại k câu tốt nhất
                sequences = sorted(all_candidates, key=lambda x: x[1], reverse=True)[:beam_size]

                # Nếu tất cả đã kết thúc bằng <eos> thì dừng sớm
                if all(tokenizer["idx2word"][seq[-1]] == '<eos>' for seq, _, _ in sequences):
                    break

            # Chọn caption tốt nhất
            best_seq = sequences[0][0][1:]  # Bỏ <bos>
            caption = []
            for idx in best_seq:
                word = tokenizer["idx2word"][idx]
                if word == '<eos>': break
                caption.append(word)

            return ' '.join(caption)