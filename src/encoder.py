from math import sqrt

import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import PositionWiseFeedForwardNetwork
from .positional_encoding import PositionalEncoding


class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, drop_rate):
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(d_model, num_heads, drop_rate)
        self.feed_forward = PositionWiseFeedForwardNetwork(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x, mask):
        # Pre-norm self-attention sublayer with residual connection
        x_pre_normed = self.norm1(x)
        x_mlha, _ = self.multi_head_attention(x_pre_normed, x_pre_normed, x_pre_normed, mask)
        x = x + self.dropout(x_mlha)

        # Pre-norm feed-forward sublayer with residual connection
        x_pre_normed = self.norm2(x)
        ffn_out = self.feed_forward(x_pre_normed)
        x = x + self.dropout(ffn_out)

        return x


class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, drop_rate, max_len):
        super().__init__()
        self.N = N
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len, drop_rate)

        self.layers = nn.ModuleList([
            EncoderLayer(d_model, heads, d_ff, drop_rate) for _ in range(N)
        ])
        self.norm = nn.LayerNorm(d_model)

        self.d_model = d_model

    def forward(self, x, mask):
        x = self.embedding(x) * sqrt(self.d_model)
        x = self.pe(x)

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)
