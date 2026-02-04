import torch
import torch.nn.functional as F
import math

# 假设 1 个 Query，3 个 Key，每个向量维度 4
Q = torch.randn(1, 4)      # (1, d)
K = torch.randn(3, 4)      # (3, d)
V = torch.tensor([[10.0], [5.0], [2.0]])  # Value

# 1. 点积相似度
scores = torch.matmul(Q, K.T) / math.sqrt(4)

# 2. softmax → 注意力
attn = F.softmax(scores, dim=-1)

# 3. 加权求和
output = torch.matmul(attn, V)

print("attention:", attn)
print("output:", output)
