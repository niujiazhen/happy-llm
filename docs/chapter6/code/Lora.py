# Lora_fix.py
import os
import json
from typing import Iterator, Dict, List
import torch
from torch.utils.data import IterableDataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, PeftModel

# -------- 配置区（按需修改） --------
MODEL_PATH = r"autodl-tmp\qwen-1.5b"   # Windows 路径用 raw string 或正斜杠
TRAIN_FILE = r"autodl-tmp\dataset\sft_data\train_3.5M_CN.json"  # 支持 jsonl 或 json array
OUTPUT_DIR = "./lora_out"
# ------------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

# 1) 加载 tokenizer 与 base model
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, trust_remote_code=True, device_map="auto")
# 若你没有多卡或 device_map 报错，可改为 model.to(device)

# 2) LoRA 配置
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj"]  # 如模型命名不同需调整
)
model = get_peft_model(model, peft_config)

# 3) 自定义 IterableDataset（逐行读入，节省内存）
class JsonlIterableDataset(IterableDataset):
    def __init__(self, path: str, tokenizer, max_length: int = 1024):
        self.path = path
        self.tokenizer = tokenizer
        self.max_length = max_length

    def parse_file(self):
        # 支持 jsonl（每行 JSON）或 JSON array（整个文件是 list）
        with open(self.path, "r", encoding="utf-8") as f:
            first = f.readline().strip()
            if not first:
                return
            # 判断是不是 jsonl：尝试看每行是否单独是 json object（如果首行以 '[' 开头则是 array）
            if first.startswith("["):
                # 整体 JSON 数组
                f.seek(0)
                data = json.load(f)
                for obj in data:
                    yield obj
            else:
                # jsonl：先行已经读了首行
                try:
                    obj = json.loads(first)
                    yield obj
                except:
                    # 若首行不是 json，可尝试跳过
                    pass
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except:
                        continue

    def __iter__(self) -> Iterator[Dict]:
        for item in self.parse_file():
            # 这里假定每个 item 有 "text" 字段或 "prompt" 字段或 "input" 字段
            # 你可以根据你的数据格式调整下面的选取逻辑
            text = None
            if isinstance(item, dict):
                for key in ("text", "content", "prompt", "input"):
                    if key in item:
                        text = item[key]
                        break
                # 如果是 SFT 样式的 { "instruction":..., "input":..., "output":... } 可拼接
                if text is None and "instruction" in item and "output" in item:
                    text = item["instruction"] + "\n" + item.get("input","") + "\n" + item["output"]
            else:
                text = str(item)

            if not text:
                continue

            enc = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding=False  # padding 在 DataCollator 里处理
            )
            input_ids = enc["input_ids"]
            yield {"input_ids": input_ids, "labels": input_ids}

# 4) Data collator（负责 padding 到 batch）
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False  # causal LM
)

train_dataset = JsonlIterableDataset(TRAIN_FILE, tokenizer, max_length=1024)

# 5) TrainingArguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,   # 根据显存调整，Qwen-1.5B 大模型需小 batch
    gradient_accumulation_steps=8,   # 使有效 batch = 16,32 等
    num_train_epochs=3,
    learning_rate=3e-4,
    fp16=True,
    logging_steps=10,
    save_total_limit=2,
    save_steps=200,
    remove_unused_columns=False,     # 对 IterableDataset 保持 False 比较稳妥
    dataloader_pin_memory=False
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer
)

# 6) 训练
trainer.train()

# 7) 保存 LoRA adapter
model.save_pretrained("./lora_adapter")
print("Saved LoRA adapter to ./lora_adapter")