# Encoder
import torchvision.models as models
from torch import nn
import torch

class AudioEncoder(nn.Module):
    def __init__(self, output_dim=512):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])  # Bỏ FC
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, output_dim)
        #self.projection = nn.Linear(512, 256) 

    def forward(self, x):
        # Input x: (batch_size, 1, n_mels, time) → thêm 1 channel
        x = self.features(x)
        x = self.pool(x).squeeze(-1).squeeze(-1)
        x = self.fc(x)
        return x  # (batch_size, output_dim)


# Decoder

class CaptionDecoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, encoder_dim, hidden_dim, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + encoder_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, features, captions):
        # captions: (batch_size, seq_len)
        embeddings = self.embed(captions)
        
        features_expanded = features.unsqueeze(1).repeat(1, embeddings.size(1), 1)
        inputs = torch.cat((features_expanded, embeddings), dim=2)
        #inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        hiddens, _ = self.lstm(inputs)
        outputs = self.fc(hiddens)
        return outputs  # (batch_size, seq_len+1, vocab_size)

# Combined Model
class AudioCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, audio_spec, captions):
        features = self.encoder(audio_spec)
        outputs = self.decoder(features, captions)
        return outputs