import torch
import torch.nn.functional as F
import math

# 假设 1 个句子，3 个 token，每个 token 2 维
X = torch.tensor([
    [1.0, 0.0],   # token 1
    [0.9, 0.1],   # token 2（和 token 1 很像）
    [0.0, 1.0]    # token 3（完全不同）
])

# self-attention
def self_attention(x):
    Q = x
    K = x
    V = x
    d_k = x.size(-1)
    scores = Q @ K.T / math.sqrt(d_k)
    attn = F.softmax(scores, dim=-1)
    out = attn @ V
    return out, attn

out, attn = self_attention(X)

print("Attention Matrix:\n", attn) # Attention Matrix 的每一行 = 一个 token 在“看谁”
print("Output:\n", out) # Output = 当前 token 把它关注的 token 的 Value 做加权平均后的结果
