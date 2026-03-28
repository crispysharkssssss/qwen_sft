"""
02_train_sft.py
Qwen2.5-1.5B-Instruct 的 SFT 微调脚本
环境：Windows + RTX + TRL 0.26
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
import matplotlib.pyplot as plt

# =========================
# 0. 基础环境设置
# =========================
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

MODEL_NAME = "E:/models/Qwen2.5/Qwen/Qwen2___5-1___5B-Instruct"
DATA_PATH = "medical_sft_data.json"
OUTPUT_DIR = "./output/sft_adapter"

# =========================
# 1. BitsAndBytes 4-bit 配置（强制 FP16）
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,  
)


# =========================
# 2. Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True,
    padding_side="right",
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================
# 3. 加载 4-bit 模型
# =========================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,   
    device_map="auto",
)


# 关 cache
model.config.use_cache = False

# =========================
# 4. LoRA 配置
# =========================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# =========================
# 5. 加载并整理数据
# =========================
dataset = load_dataset(
    "json",
    data_files={"train": DATA_PATH},
    split="train",
)

def build_text(example):
    text = example["instruction"]
    if example.get("input"):
        text += "\n" + example["input"]
    text += "\n### Response:\n" + example["output"]
    return {"text": text}

dataset = dataset.map(
    build_text,
    remove_columns=dataset.column_names,
)

# =========================
# 6. SFTConfig
# =========================
sft_config = SFTConfig(
    output_dir="./output/sft_adapter",

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,

    fp16=False,      
    bf16=False,     

    num_train_epochs=2,
    learning_rate=1e-4,
    logging_steps=10,
    report_to=[],

    # max_seq_length=768,
    packing=False,

    max_grad_norm=0.0,  
    optim="adamw_torch",
)


print("==== Precision Check ====")
print("fp16:", sft_config.fp16)
print("bf16:", sft_config.bf16)
print("=========================")

# =========================
# 7. 初始化 SFTTrainer
# =========================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    args=sft_config,
    processing_class=tokenizer,
)

# =========================
# 8. 开始训练
# =========================
trainer.train()

# =========================
# 绘制 Loss 曲线
# =========================

log_history = trainer.state.log_history

loss_steps = []
loss_values = []

for log in log_history:
    if "loss" in log:
        loss_steps.append(log.get("step", len(loss_steps)))
        loss_values.append(log["loss"])

if len(loss_values) > 0:
    plt.figure(figsize=(8, 5))
    plt.plot(loss_steps, loss_values)
    plt.xlabel("Step")
    plt.ylabel("Training Loss")
    plt.title("SFT Training Loss Curve")
    plt.grid(True)

    loss_path = os.path.join(sft_config.output_dir, "training_loss.png")
    plt.savefig(loss_path, dpi=150)
    plt.close()

    print(f"Loss 曲线已保存至: {loss_path}")
else:
    print("未记录到 loss，可能 logging_steps 过大")

# =========================
# 9. 保存 LoRA Adapter
# =========================
trainer.save_model(OUTPUT_DIR)

print(f"\n训练完成，LoRA Adapter 已保存到：{OUTPUT_DIR}")
