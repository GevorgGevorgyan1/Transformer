import torch.nn as nn
import torch.nn.functional as F


class PositionWiseFeedForwardNetwork(nn.Module):
    def __init__(self, d_model=512, d_ff=2048, drop_rate=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
