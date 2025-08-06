import torch
from preprocessing import preprocess_audio, generate_caption, generate_caption_deploy, generate_caption_optimized
from load_model import load_model, load_tokenizer, load_tokenizer_json
from simpletokenizer import SimpleTokenizer

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load tokenizer
    #tokenizer = Tokenizer.load("tokenizer.pkl")
    model_path = "D:/Project/Speech_Recognize/model_weights/resnet50_lstm optimizer.pth"
    tokenizer_path = "D:/Project/Speech_Recognize/model_weights/tokenizer.pkl"
    #tokenizer_path = "D:/Project/Speech_Recognize/model_weights/word2idx.json"
    tokenizer = load_tokenizer(tokenizer_path)
    model = load_model(model_path, tokenizer, device)
    
    filepath = "D:/Project/Speech_Recognize/data/plastic-95763.mp3"
    mel = preprocess_audio(filepath)

    caption = generate_caption_optimized(model, mel, tokenizer, device=device)
    print("Generated Caption:", caption)