import math

import torch
import torch.nn as nn


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    attention_weights = torch.softmax(scores, dim=-1)

    if dropout is not None:
        attention_weights = dropout(attention_weights)

    output = torch.matmul(attention_weights, v)

    return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, drop_rate=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.dropout = nn.Dropout(drop_rate)

        # Learnable projections
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.W_v = nn.Linear(d_model, d_model, bias=False)
        self.W_o = nn.Linear(d_model, d_model, bias=False)

    def forward(self, q, k, v, mask=None):
        # Project
        Q = self.W_q(q)
        K = self.W_k(k)
        V = self.W_v(v)

        # Split into heads
        q, k, v = self.split(Q), self.split(K), self.split(V)

        # Attention on each head
        x, weights = scaled_dot_product_attention(q, k, v, mask, dropout=self.dropout)

        # Concat ([batch_size, num_heads, seq_len, d_k] -> [batch_size, seq_len, d_model])
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

        # Apply output projection
        x = self.W_o(x)

        return x, weights

    def split(self, tensor):
        """Split [batch_size, seq_len, d_model] -> [batch_size, num_heads, seq_len, d_k]."""
        batch_size, seq_len, _ = tensor.shape
        tensor = tensor.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        return tensor
