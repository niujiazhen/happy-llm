from transformers import PretrainedConfig

class ModelConfig(PretrainedConfig):
    model_type = "Tiny-K"
    def __init__(
            self,
            dim: int = 768, # 模型维度: 768表示每个token的表示维度
            n_layers: int = 12, # Transformer的层数：12层表示模型有12个Transformer块，每个块包含一个多头自注意力层和一个前馈神经网络层
            n_heads: int = 16, # 注意力机制的头数：16表示每个Transformer块中的多头自注意力层有16个头，每个头可以独立地关注输入序列的不同部分，从而捕捉更多的上下文信息
            n_kv_heads: int = 8, # 键值头的数量：8表示在多头自注意力层中，键和值的头数为8，这通常用于减少计算复杂度，同时保持模型的表达能力
            vocab_size: int = 6144, # 词汇表大小：6144表示模型可以处理的不同token的数量，这个值通常取决于训练数据中使用的词汇表大小
            hidden_dim: int = None, # 隐藏层维度：hidden_dim表示Transformer块中前馈神经网络的隐藏层维度，通常是模型维度的4倍，例如如果dim=768，则hidden_dim可以设置为3072
            multiple_of: int = 64, 
            norm_eps: float = 1e-5, # 归一化层的eps
            max_seq_len: int = 512, # 最大序列长度
            dropout: float = 0.0, # dropout概率
            flash_attn: bool = True, # 是否使用Flash Attention
            **kwargs,
    ):
        self.dim = dim
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.flash_attn = flash_attn
        super().__init__(**kwargs)

if __name__ == "__main__":
    config = ModelConfig()
    print(config)