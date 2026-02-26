# 加载定义好的模型参数-此处以 Qwen-2.5-1.5B 为例
# 使用 transforemrs 的 Config 类进行加载
from transformers import AutoConfig

model_path = "autodl-tmp/qwen-1.5b"
config = AutoConfig.from_pretrained(model_path)
print(config)