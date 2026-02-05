import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFFN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * 4)
        self.fc2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))

x = torch.randn(2, 5, 8)  # (batch, seq_len, hidden)
ffn = SimpleFFN(8)

y = ffn(x)
print(y.shape)

# FNN不会混token
