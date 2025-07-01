import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer

class PianoRollTransformer(nn.Module):
    def __init__(self,
                 text_vocab_size,
                 structured_feature_dim, # Dimension of your combined structured features
                 max_seq_len,
                 num_patches, # Number of time steps in your piano roll
                 num_pitches, # Number of MIDI pitches (e.g., 128)
                 embedding_dim,
                 nhead,
                 num_encoder_layers,
                 num_decoder_layers,
                 dropout=0.1):
        super().__init__()

        self.text_embedding = nn.Embedding(text_vocab_size, embedding_dim)
        self.structured_embedding = nn.Linear(structured_feature_dim, embedding_dim)
        self.positional_encoding = PositionalEncoding(embedding_dim, dropout, max_seq_len)

        encoder_layers = TransformerEncoderLayer(embedding_dim, nhead, dim_feedforward=4*embedding_dim, dropout=dropout, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_encoder_layers)

        self.pitch_embedding = nn.Embedding(num_pitches + 1, embedding_dim) # +1 for the 'no note' token
        self.time_embedding = nn.Embedding(num_patches, embedding_dim) # Embedding for each time step

        decoder_layers = TransformerDecoderLayer(embedding_dim, nhead, dim_feedforward=4*embedding_dim, dropout=dropout, batch_first=True)
        self.transformer_decoder = TransformerDecoder(decoder_layers, num_decoder_layers)

        self.fc_pitch = nn.Linear(embedding_dim, num_pitches + 1) # Predict pitch (including 'no note')
        self.fc_velocity = nn.Linear(embedding_dim, 1) # Predict velocity (you might need to adjust the output range/quantization)

        self.num_patches = num_patches
        self.num_pitches = num_pitches
        self.embedding_dim = embedding_dim

    def forward(self, src_text, structured_features, tgt_pianoroll):
        # Encoder
        src_embedded = self.positional_encoding(self.text_embedding(src_text) + self.structured_embedding(structured_features))
        memory = self.transformer_encoder(src_embedded)

        # Decoder
        batch_size, seq_len, _ = tgt_pianoroll.shape
        time_steps = torch.arange(seq_len, device=tgt_pianoroll.device).unsqueeze(0).repeat(batch_size, 1)
        tgt_pitch = tgt_pianoroll[:, :, 0].long() # Assuming pitch is the first element
        tgt_velocity = tgt_pianoroll[:, :, 1].unsqueeze(-1) # Assuming velocity is the second

        tgt_embedded = self.positional_encoding(self.pitch_embedding(tgt_pitch) + self.time_embedding(time_steps))
        decoder_output = self.transformer_decoder(tgt_embedded, memory)

        pitch_prediction = self.fc_pitch(decoder_output)
        velocity_prediction = torch.sigmoid(self.fc_velocity(decoder_output)) # Assuming velocity is normalized to [0, 1]

        return pitch_prediction, velocity_prediction

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)