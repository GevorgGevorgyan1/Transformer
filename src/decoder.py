from math import sqrt

import torch.nn as nn

from .attention import MultiHeadAttention
from .feed_forward import PositionWiseFeedForwardNetwork
from .positional_encoding import PositionalEncoding


class DecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, drop_rate):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(drop_rate)

        self.masked_self_attention = MultiHeadAttention(d_model, num_heads, drop_rate)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, drop_rate)

        self.ffn = PositionWiseFeedForwardNetwork(d_model, d_ff)

    def forward(self, x, encoder_output, self_attn_mask, cross_attn_mask):
        """
        Args:
            x:               Decoder input — (batch, target_seq_len, d_model)
            encoder_output:  Output from the encoder — (batch, src_seq_len, d_model)
            self_attn_mask:  Causal mask for decoder self-attention — prevents attending to future tokens
            cross_attn_mask: Padding mask from encoder — prevents attending to encoder padding tokens
        """
        # Pre-norm masked self-attention with residual connection
        x_normed = self.norm1(x)
        self_attn_out, _ = self.masked_self_attention(x_normed, x_normed, x_normed, self_attn_mask)
        x = x + self.dropout(self_attn_out)

        # Pre-norm cross-attention with residual connection
        x_normed = self.norm2(x)
        cross_attn_out, _ = self.cross_attention(x_normed, encoder_output, encoder_output, cross_attn_mask)
        x = x + self.dropout(cross_attn_out)

        # Pre-norm feed-forward with residual connection
        x_normed = self.norm3(x)
        ffn_out = self.ffn(x_normed)
        x = x + self.dropout(ffn_out)

        return x


class Decoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, num_heads, d_ff, drop_rate, max_len):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, max_len, drop_rate)
        self.layers = nn.ModuleList([
            DecoderLayer(d_model, num_heads, d_ff, drop_rate) for _ in range(N)
        ])

        self.norm = nn.LayerNorm(d_model)

        self.d_model = d_model

    def forward(self, x, encoder_output, self_attn_mask, cross_attn_mask):
        x = self.embedding(x) * sqrt(self.d_model)
        x = self.pe(x)

        for layer in self.layers:
            x = layer(x, encoder_output, self_attn_mask, cross_attn_mask)

        return self.norm(x)
