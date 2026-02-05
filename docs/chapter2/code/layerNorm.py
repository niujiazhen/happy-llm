import torch
import torch.nn as nn

x = torch.randn(2, 4, 6)  # (batch, seq_len, hidden)
ln = nn.LayerNorm(6)

y = ln(x)

# 看一个 token
print(x[0, 0].mean(), x[0, 0].std())
print(y[0, 0].mean(), y[0, 0].std())
