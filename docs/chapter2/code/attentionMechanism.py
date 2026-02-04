import torch
import torch.nn.functional as F
import math
'''注意力计算函数'''
def attention(query, key, value, dropout=None):
    '''
    args:
    query: 查询值矩阵
    key: 键值矩阵
    value: 真值矩阵
    '''
    # 获取键向量的维度，键向量的维度和值向量的维度相同
    d_k = query.size(-1) 
    # 计算Q与K的内积并除以根号dk
    # transpose——相当于转置
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)# 计算相似度
    # Softmax
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
        # 采样
     # 根据计算结果对value进行加权求和
    return torch.matmul(p_attn, value), p_attn


# Query：我们关心的“概念”
Q = torch.tensor([[1.0, 0.0]])   # 偏向第 1 个维度

# 3 个 Key
K = torch.tensor([
    [1.0, 0.0],   # 和 Query 几乎一样（非常相关）
    [0.5, 0.5],   # 有点相关
    [0.0, 1.0]    # 完全不相关
])

# Value 是“真正的信息”
V = torch.tensor([
    [10.0],   # 对应第 1 个 Key
    [5.0],    # 对应第 2 个 Key
    [1.0]     # 对应第 3 个 Key
])

output, attn = attention(Q, K, V)

print("Attention 权重:", attn)
print("最终输出:", output)

Q2 = torch.tensor([[0.0, 1.0]])  # 偏向第 2 个维度
output2, attn2 = attention(Q2, K, V)

print("Attention 权重:", attn2)
print("最终输出:", output2)


