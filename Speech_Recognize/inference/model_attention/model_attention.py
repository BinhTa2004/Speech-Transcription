import torch
import torch.nn as nn
import torchvision.models as models

class AudioEncoder_Attention(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]  # Bỏ global avgpool + fc
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((10, 10))  # Có thể thêm pooling nhẹ nếu cần giới hạn time_steps

    def forward(self, x):
        x = self.resnet(x)  # (batch_size, 2048, H, W), thường H=W=~7
        x = self.adaptive_pool(x)  # (batch_size, 2048, 10, 10) để giảm kích thước sequence
        x = x.permute(0, 2, 3, 1)  # (batch_size, 10, 10, 2048)
        x = x.view(x.size(0), -1, 2048)  # (batch_size, 100, 2048) --> sequence cho attention
        return x  # output shape: (batch_size, time_steps, encoder_dim)


import torch
import torch.nn as nn
import torch.nn.functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)

    def forward(self, encoder_outputs, decoder_hidden):
        # encoder_outputs: (batch_size, time_steps, encoder_dim)
        # decoder_hidden: (batch_size, decoder_dim)

        # Add time dimension to decoder hidden
        decoder_hidden = decoder_hidden.unsqueeze(1)  # (batch_size, 1, decoder_dim)

        att1 = self.encoder_att(encoder_outputs)  # (batch_size, time_steps, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # .unsqueeze(1)  # (batch_size, 1, attention_dim)

        att = torch.tanh(att1 + att2)  # broadcast add
        e = self.full_att(att).squeeze(2)  # (batch_size, time_steps)

        alpha = F.softmax(e, dim=1)  # attention weights (batch_size, time_steps)

        context = (encoder_outputs * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return context, alpha


class CaptionDecoder_Attention(nn.Module):
    def __init__(self, vocab_size, embed_dim, decoder_dim, hidden_dim, attention_module, num_layers=1):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim + decoder_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self.attention = attention_module  # <-- attention module đã định nghĩa ở trên

    def forward(self, encoder_outputs, captions):
        batch_size, seq_len = captions.shape
        embeddings = self.embed(captions)  # (batch_size, seq_len, embed_dim)

        h = torch.zeros(1, batch_size, self.lstm.hidden_size).to(captions.device)
        c = torch.zeros(1, batch_size, self.lstm.hidden_size).to(captions.device)

        outputs = []
        for t in range(seq_len):
            embedding_t = embeddings[:, t, :]  # (batch, embed_dim)

            if t == 0:
                # Khởi tạo attention bằng 0
                context = torch.zeros(batch_size, encoder_outputs.shape[2]).to(captions.device)
            else:
                # decoder_hidden = h[-1]
                # context, _ = self.attention(encoder_outputs, decoder_hidden)
                context, _ = self.attention(encoder_outputs, h.squeeze(0))  # (batch, encoder_dim)

            lstm_input = torch.cat((embedding_t, context), dim=1).unsqueeze(1)
            output, (h, c) = self.lstm(lstm_input, (h, c))
            output_vocab = self.fc(output.squeeze(1))
            outputs.append(output_vocab)

        outputs = torch.stack(outputs, dim=1)  # (batch_size, seq_len, vocab_size)
        return outputs

class AudioCaptioningModel(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, audio_spec, captions):
        features = self.encoder(audio_spec)
        outputs = self.decoder(features, captions)
        return outputs