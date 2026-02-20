import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

import ModelConfig
from RMSNorm import RMSNorm
import GQA
from MLP import MLP

class DecoderLayer(nn.Module):
    def __init__(self, layer_id: int, args: ModelConfig.ModelConfig):
        super().__init__()

        # define the attention layer
        self.n_head = args.n_heads

        # define the input dimensions
        self.dim = args.dim

        # define the dimension of each head
        self.head_dim = self.dim // self.n_head

        # define the GQA attention layer
        self.attention = GQA.Attention(args)

        # define the MLP layer
        self.feed_forward = MLP(args.dim, args.hidden_dim, args.multiple_of, args.dropout)

        # define layer id
        self.layer_id = layer_id

        # define RMSNorm layer for GQA
        self.attention_norm=RMSNorm(args.dim, eps=args.norm_eps)

        # define RMSNorm layer for MLP
        self.ffn_norm=RMSNorm(args.dim, eps=args.norm_eps)

    def forward(self, x: torch.Tensor, freqs_cos, freqs_sin) -> torch.Tensor:
        # x的形状为[batch_size, seq_len, dim]
        # freqs_cos和freqs_sin的形状为[seq_len, head_dim]

        # 1. GQA attention
        # 首先对输入x进行RMSNorm归一化，然后将结果传入GQA注意力层，得到注意力输出attn_out
        attn_out = self.attention(self.attention_norm(x), freqs_cos, freqs_sin)

        # 2. Add & Norm
        # 将输入x与注意力输出attn_out相加，得到残差连接的结果residual1
        residual1 = x + attn_out

        # 3. MLP
        # 将残差连接的结果residual1进行RMSNorm归一化，然后传入MLP层，得到MLP输出ffn_out
        ffn_out = self.feed_forward(self.ffn_norm(residual1))

        # 4. Add & Norm
        # 将残差连接的结果residual1与MLP输出ffn_out相加，得到最终的输出output
        output = residual1 + ffn_out

        return output

if __name__ == "__main__":
    args=ModelConfig.ModelConfig()
    # 创建LLaMADecoderLayer实例
    decoderlayer = DecoderLayer(0, args)

    # 模拟输入数据
    dim = args.dim
    seq_len = 50

    x = torch.randn(1, seq_len, dim) # [bs, seq_len, dim]

    freqs_cos, freqs_sin = GQA.precompute_freqs_cis(dim//args.n_heads, seq_len)

    out = decoderlayer(x, freqs_cos, freqs_sin)

    print(out.shape) # 形状和输入的x一样 [batch_size, seq_len, dim]