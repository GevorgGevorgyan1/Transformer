import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, drop_rate=0.1):
        super().__init__()
        pe = torch.zeros(max_len, d_model)

        self.dropout = nn.Dropout(drop_rate)

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        if x.size(1) > self.pe.size(1):
            raise ValueError(f"Sequence length {x.size(1)} exceeds max_len {self.pe.size(1)}")
        x = x + self.pe[:, :x.shape[1], :]
        return self.dropout(x)
