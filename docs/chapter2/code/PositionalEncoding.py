import torch
import math
import torch.nn.functional as F
import torch.nn as nn
class PositionalEncoding(nn.Module):
    '''位置编码模块'''

    def __init__(self, args):
        super(PositionalEncoding, self).__init__()
        # Dropout 层
        # self.dropout = nn.Dropout(p=args.dropout)

        # block size 是序列的最大长度
        pe = torch.zeros(args.block_size, args.n_embd)
        position = torch.arange(0, args.block_size).unsqueeze(1)
        # 计算 theta
        div_term = torch.exp(
            torch.arange(0, args.n_embd, 2) * -(math.log(10000.0) / args.n_embd)
        )
        # 分别计算 sin、cos 结果
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 将位置编码加到 Embedding 结果上
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return x
    
class Args:
    block_size = 20
    n_embd = 16

args = Args()
pe_layer = PositionalEncoding(args)

# 取位置编码（不经过 embedding）
pe = pe_layer.pe[0]  # shape: [block_size, d_model]

# 计算相似度
sim_01 = F.cosine_similarity(pe[0], pe[1], dim=0)
sim_02 = F.cosine_similarity(pe[0], pe[2], dim=0)
sim_010 = F.cosine_similarity(pe[0], pe[10], dim=0)

print("sim(0,1):", sim_01.item())
print("sim(0,2):", sim_02.item())
print("sim(0,10):", sim_010.item())