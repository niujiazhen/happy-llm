import json
import random
import re
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import os

"""
Pretrain Dataset 主要是将 `text` 通过 `tokenizer` 转换成 `input_id`，
然后将 `input_id` 拆分成 `X` 和 `Y`，
其中 `X` 为 `input_id` 的前 n-1 个元素，`Y` 为 `input_id` 的后 n-1 `个元素。
loss_mask` 主要是用来标记哪些位置需要计算损失，哪些位置不需要计算损失

Pretrain：
1、所有 token 都参与 loss
2、自回归：预测下一个 token
3、无角色区分
"""
class PretrainDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        # 预计算每行的起始字节偏移量
        self._offsets = []
        with open(data_path, 'rb') as f:
            self._offsets.append(0)
            while f.readline():
                self._offsets.append(f.tell())
        self._total_lines = len(self._offsets) - 1  # 最后一个 tell() 是 EOF

    def __len__(self):
        return self._total_lines

    def __getitem__(self, index: int):
        with open(self.data_path, 'rb') as f:
            f.seek(self._offsets[index])
            line = f.readline().decode('utf-8')
        sample = json.loads(line)
        text = f"{self.tokenizer.bos_token}{sample['text']}" # 在文本前添加 BOS 标记,以便模型知道文本的开始位置`
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length] # 将文本编码为输入 ID，并截断到最大长度
        text_len = len(input_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - text_len # 计算需要填充的长度
        input_id = input_id + [self.padding] * padding_len
        # 0表示不计算损失
        loss_mask = [1] * text_len + [0] * padding_len # 创建损失掩码，文本部分为 1，填充部分为 0

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64) # 输入 ID 的前 n-1 个作为模型输入
        Y = np.array(input_id[1:]).astype(np.int64) # 输入 ID 的后 n-1 个作为模型的目标输出
        loss_mask = np.array(loss_mask[1:]).astype(np.int64) # 损失掩码的后 n-1 个与输入 ID 对齐，指示哪些位置需要计算损失
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)# 将 NumPy 数组转换为 PyTorch 张量并返回输入、目标输出和损失掩码
    

class SFTDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512):
        super().__init__()
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = 0
        self._offsets = []
        with open(data_path, 'rb') as f:
            self._offsets.append(0)
            while f.readline():
                self._offsets.append(f.tell())
        self._total_lines = len(self._offsets) - 1 

    def __len__(self):
        return self._total_lines

    def generate_loss_mask(self, input_ids):
        # 生成 loss mask, 0 表示不计算损失, 1 表示计算损失
        mask = [0] * len(input_ids)
        # 当遇到 `|<im_start|>assistant\n` 时，就开始计算损失，直到遇到 `|<im_end|>` 为止
        a_sequence = self.tokenizer("<|im_start|>assistant\n")['input_ids']  # <|im_start|>assistant\n
        a_length = len(a_sequence)
        n = len(input_ids)
        i = 0
        
        while i <= n - a_length:
            # 检查当前位置是否匹配目标子序列
            match = True
            for k in range(a_length):
                if input_ids[i + k] != a_sequence[k]:
                    match = False
                    break
            if match:
                # 从子序列结束的位置开始查找第一个 4 (eos_token_id)
                j = None
                for idx in range(i + a_length, n):
                    if input_ids[idx] == self.tokenizer.eos_token_id:
                        j = idx
                        break
                if j is not None:
                    start = i + a_length
                    end = j  # 结束位置设为j（包含4）
                    # 标记区间为1（包括start到end）
                    if start <= end:
                        for pos in range(start, end + 1):
                            if pos < len(mask):
                                mask[pos] = 1
                # 跳过当前子序列，避免重叠匹配
                i += a_length
            else:
                i += 1
        return mask

    def __getitem__(self, index: int):
        with open(self.data_path, 'rb') as f:
            f.seek(self._offsets[index])
            line = f.readline().decode('utf-8')
        sample = json.loads(line)
        text = self.tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=False)
        input_id = self.tokenizer(text).data['input_ids'][:self.max_length]
        text_len = len(input_id)
        # 没满最大长度的剩余部分
        padding_len = self.max_length - text_len
        input_id = input_id + [self.padding] * padding_len
        # 0表示不计算损失
        loss_mask = self.generate_loss_mask(input_id)

        input_id = np.array(input_id)
        X = np.array(input_id[:-1]).astype(np.int64)
        Y = np.array(input_id[1:]).astype(np.int64)
        loss_mask = np.array(loss_mask[1:]).astype(np.int64)
        return torch.from_numpy(X), torch.from_numpy(Y), torch.from_numpy(loss_mask)