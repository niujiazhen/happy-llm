import torch
import torch.nn as nn

# 假设
vocab_size = 10
d_model = 8
seq_len = 4
batch_size = 2

# Embedding 层
embedding = nn.Embedding(vocab_size, d_model)

# 假装这是 token ids
x = torch.tensor([
    [1, 3, 5, 7],
    [2, 4, 6, 8]
])  # shape: (batch, seq_len)

out = embedding(x)

print("input shape:", x.shape)
print("output shape:", out.shape)
print(out)

x1 = torch.tensor([[1, 3, 5, 7]])
x2 = torch.tensor([[7, 5, 3, 1]])

e1 = embedding(x1)
e2 = embedding(x2)

print(torch.allclose(e1[:, 0], e2[:, -1]))
