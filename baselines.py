# baselines.py
import torch
import torch.nn as nn
import math

class TransformerBaseline(nn.Module):
    def __init__(self, dim=128, depth=3, vocab_size=2048, max_len=2049, heads=4, num_classes=32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.pos_emb = nn.Embedding(max_len, dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=dim*4)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.pool = nn.Linear(dim, num_classes)
        self.dim = dim

    def forward(self, x):
        # x: (B, T)
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = self.token_emb(x) + self.pos_emb(pos)
        # Transformer expects (T, B, dim)
        h = h.transpose(0, 1)
        h = self.encoder(h)            # (T, B, dim)
        h = h.transpose(0, 1)          # (B, T, dim)
        # use final position pooling like your task queries the last token
        last = h[:, -1, :]             # (B, dim)
        logits = self.pool(last)
        # return similar signature to SynapNet: logits, None, None, None
        return logits, None, None, None

class SSMProxy(nn.Module):
    """
    Simple SSM-like proxy: a small stack of causal 1D convs + gating or an LSTM/GRU.
    This is not Mamba but is a fair SSM-style baseline for comparison.
    """
    def __init__(self, dim=128, depth=3, vocab_size=2048, num_classes=32):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, dim)
        self.rnn = nn.GRU(input_size=dim, hidden_size=dim, num_layers=depth, batch_first=True)
        self.pool = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x: (B, T)
        h = self.token_emb(x)       # (B, T, dim)
        out, _ = self.rnn(h)        # (B, T, dim)
        last = out[:, -1, :]        # (B, dim)
        logits = self.pool(last)
        return logits, None, None, None
