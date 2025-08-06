import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import numpy as np
import os

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from early_stopping import EarlyStopping
from utils import AudioCaptionDataset, SimpleTokenizer
from model import AudioCaptioningModel, AudioEncoder, CaptionDecoder


df = pd.read_csv(r"/kaggle/input/clotho-dataset/clotho_captions_development.csv")
audio_folder = r'/kaggle/input/clotho-dataset/clotho_audio_development'

def find_audio_path(file_name):
    path = os.path.join(audio_folder, file_name)
    if os.path.exists(path):
        return path
    # Nếu không tìm thấy file nào, trả về None hoặc mặc định đuôi .jpg
    print("not find",path)

df['audio_path'] = df['file_name'].apply(find_audio_path)
df.head()

data = []
for _, row in df.iterrows():
    audio_path = row["audio_path"]
    captions = [row[f"caption_{i}"] for i in range(1, 6)]

    for caption in captions:
        data.append((audio_path, caption))
audio_paths, captions = zip(*data)

train_audio_paths, val_audio_paths, train_captions, val_captions = train_test_split(audio_paths, captions, test_size=0.1, random_state=42)

all_captions = list(train_captions) + list(val_captions)
tokenizer = SimpleTokenizer(all_captions)

train_dataset = AudioCaptionDataset(train_audio_paths, train_captions, tokenizer)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4) #, collate_fn=collate_fn

val_dataset = AudioCaptionDataset(val_audio_paths, val_captions, tokenizer)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=True, num_workers=4) #, collate_fn=collate_fn                                                                                    test_size=0.1, random_state=42)
# Training
def train_one_epoch(model, train_loader, optimizer, criterion, device):
    model.train()
    total_loss, total_samples = 0, 0
    loop = tqdm(train_loader, leave=True)
    for mel, captions in loop:
        mel = mel.to(device)              
        captions = captions.to(device)  

        optimizer.zero_grad()
        outputs = model(mel, captions[:, :-1])  
        loss = criterion(outputs.reshape(-1, outputs.size(-1)), captions[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()

        batch_size = mel.size(0)  
        total_loss += loss.item() * batch_size  
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    return avg_loss

# Validation
def validate(model, val_loader, criterion, device):
    model.eval()
    total_loss, total_samples = 0, 0
    with torch.no_grad():
        loop = tqdm(val_loader, leave=True)
        for mel, captions in loop:
            mel = mel.to(device)              
            captions = captions.to(device)  
        
            optimizer.zero_grad()
            outputs = model(mel, captions[:, :-1])   # Teacher Forcing: input là caption trừ token cuối
            loss = criterion(outputs.reshape(-1, outputs.size(-1)), captions[:, 1:].reshape(-1))     
            
            batch_size = mel.size(0)  # số sample trong batch
            total_loss += loss.item() * batch_size  # cộng dồn tổng loss theo sample
            total_samples += batch_size
            
    avg_loss = total_loss / total_samples
    return avg_loss 

vocab_size = tokenizer.vocab_size  
encoder = AudioEncoder(output_dim=512)
decoder = CaptionDecoder(vocab_size, embed_dim=256, encoder_dim=512, hidden_dim=512)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = AudioCaptioningModel(encoder, decoder)
model = nn.DataParallel(model).to(device)

optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

num_epochs = 20
early_stopper = EarlyStopping(patience=3, verbose=True)
for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader,  optimizer, criterion, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss:.4f}")

    val_loss = validate(model, val_loader, criterion, device)
    print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {val_loss:.4f}")
    
    early_stopper(val_loss, model)
    if early_stopper.early_stop:
        print("Early stopping")
        break

torch.save(model.state_dict(), "audio_caption_model_for_inference.pth")
print("Model saved for inference!")
