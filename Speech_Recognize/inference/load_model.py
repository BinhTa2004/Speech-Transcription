import pickle
import torch
import json

from inference.model import AudioCaptioningModel, AudioEncoder_18, CaptionDecoder, AudioEncoder_50
from inference.utils import remove_module_prefix
from inference.simpletokenizer import SimpleTokenizer

# from model import AudioCaptioningModel, AudioEncoder_18, CaptionDecoder, AudioEncoder_50
# from utils import remove_module_prefix

file_path = "D:/Project/Speech_Recognize/inference/config_resnet50.json"
with open(file_path) as f:
    config = json.load(f)

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

def load_model(model_path, tokenizer, device):
    encoder = AudioEncoder_50(output_dim=config["encoder"]["encoder_output_size"])
    decoder = CaptionDecoder(
        #vocab_size=tokenizer.vocab_size,
        vocab_size=tokenizer["vocab_size"], # Deploy thì dùng cái này
        embed_dim=config["decoder"]["embedding_dim"],
        encoder_dim=config["decoder"]["encoder_dim"],
        hidden_dim=config["decoder"]["hidden_dim"],
        num_layers=config["decoder"]["num_layers"]
    )
    model = AudioCaptioningModel(encoder, decoder).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    new_state_dict = remove_module_prefix(checkpoint)
    # Load lại vào model
    model.load_state_dict(new_state_dict)
    model.eval()

    return model

# Hàm load tokenizer mới (từ word2idx.json)
def load_tokenizer_json(word2idx_path):
    with open(word2idx_path, "r") as f:
        word2idx = json.load(f)

    idx2word = {int(idx): word for word, idx in word2idx.items()}
    vocab_size = len(word2idx)
    pad_token_id = word2idx['<pad>']

    # Viết luôn encode/decode dưới dạng closure function
    def encode(text, max_len=30):
        tokens = [word2idx.get(word, word2idx['<unk>']) for word in text.lower().strip().split()]
        tokens = [word2idx['<bos>']] + tokens + [word2idx['<eos>']]
        if len(tokens) < max_len:
            tokens += [word2idx['<pad>']] * (max_len - len(tokens))
        else:
            tokens = tokens[:max_len]
        return tokens

    def decode(tokens):
        words = [idx2word.get(idx, '<unk>') for idx in tokens]
        return ' '.join(words)

    # Trả về một dict đầy đủ
    tokenizer = {
        "word2idx": word2idx,
        "idx2word": idx2word,
        "vocab_size": vocab_size,
        "pad_token_id": pad_token_id,
        "encode": encode,
        "decode": decode
    }
    return tokenizer


