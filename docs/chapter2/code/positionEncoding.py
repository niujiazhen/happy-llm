import torch
import math

def positional_encoding(seq_len, d_model):
    pe = torch.zeros(seq_len, d_model)
    for pos in range(seq_len):
        for i in range(0, d_model, 2):
            div_term = math.pow(10000, i / d_model)
            pe[pos, i] = math.sin(pos / div_term)
            pe[pos, i + 1] = math.cos(pos / div_term)
    return pe

pe = positional_encoding(seq_len=10, d_model=8)
print(pe)
import torch.nn.functional as F

# 比较位置 0 和位置 1 的相似度
sim_01 = F.cosine_similarity(pe[0], pe[1], dim=0)
sim_09 = F.cosine_similarity(pe[0], pe[9], dim=0)

print("sim(pos0, pos1):", sim_01.item())
print("sim(pos0, pos9):", sim_09.item())
