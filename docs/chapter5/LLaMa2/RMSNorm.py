import torch
import torch.nn as nn
import ModelConfig
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        # eps是为了防止除以0的情况
        self.eps = eps
        # weight是一个可学习的参数，全部初始化为1
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        # 计算RMSNorm的核心部分
        # x.pow(2).mean(-1, keepdim=True)计算了输入x的平方的均值
        # torch.rsqrt是平方根的倒数，这样就得到了RMSNorm的分母部分，再加上eps防止分母为0
        # 最后乘以x，得到RMSNorm的结果
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        # forward函数是模型的前向传播
        # 首先将输入x转为float类型，然后进行RMSNorm，最后再转回原来的数据类型
        # 最后乘以weight，这是RMSNorm的一个可学习的缩放因子
        output = self._norm(x.float()).type_as(x)
        return output * self.weight
    
if __name__ == "__main__":
    args=ModelConfig.ModelConfig()
    norm = RMSNorm(args.dim, args.norm_eps)
    x = torch.randn(1, 50, args.dim) # 模拟一个输入，形状为[1, 50, 768]，表示一个batch中有1个序列，每个序列有50个token，每个token的表示维度是768
    output = norm(x)  
    print(output.shape)

    # out:
    # torch.Size([1, 50, 768])