import torch
import torch.nn.functional as F
import math

x = torch.randn(4, 2)  # 4 个 token

scores = x @ x.T / math.sqrt(2)

mask = torch.triu(torch.full((4, 4), float("-inf")), diagonal=1)

masked_scores = scores + mask
attn = F.softmax(masked_scores, dim=-1)

print(attn)
