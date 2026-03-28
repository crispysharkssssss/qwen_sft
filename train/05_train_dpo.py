"""
05_train_dpo.py

在 SFT 基础上进行 DPO 对齐（稳定版）
模型：Qwen2.5-1.5B-Instruct
策略：
- SFT adapter 作为 policy
- reference 由 DPOTrainer 自动创建（policy 初始快照）
- 4-bit + LoRA + 低学习率
"""

import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel
from trl import DPOTrainer, DPOConfig

# ======================================================
# 0. 基础环境
# ======================================================
os.environ["WANDB_DISABLED"] = "true"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

BASE_MODEL = "E:/models/Qwen2.5/Qwen/Qwen2___5-1___5B-Instruct"
SFT_ADAPTER = "./output/sft_adapter"
DPO_DATA = "./data/medical_dpo_data.json"  
OUTPUT_DIR = "./output/dpo_adapter"

# ======================================================
# 1. BitsAndBytes 4-bit 配置
# ======================================================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16,
)

# ======================================================
# 2. Tokenizer
# ======================================================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    padding_side="right",
)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ======================================================
# 3. 加载 base model
# ======================================================
print("🔹 Loading base model (4-bit)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto",
)
base_model.config.use_cache = False

# ======================================================
# 4. 加载 SFT adapter 作为 policy
# ======================================================
print("🔹 Loading SFT adapter as policy model...")
policy_model = PeftModel.from_pretrained(
    base_model,
    SFT_ADAPTER,
    is_trainable=True,   
)
policy_model.train()

# 自检：确保真的有可训练参数
trainable_params = sum(
    p.numel() for p in policy_model.parameters() if p.requires_grad
)
print(f"Trainable parameters: {trainable_params:,}")
assert trainable_params > 0, "没有可训练参数，DPO 会直接失败"

# ======================================================
# 5. 加载 DPO 数据
# 必须包含字段：prompt / chosen / rejected
# ======================================================
dataset = load_dataset(
    "json",
    data_files={"train": DPO_DATA},
    split="train",
)

print("DPO dataset size:", len(dataset))
print("Sample:")
print(dataset[0])

# ======================================================
# 6. DPO 配置（医疗任务 · 稳定优先）
# ======================================================
dpo_config = DPOConfig(
    output_dir=OUTPUT_DIR,

    beta=0.1,                     
    learning_rate=5e-6,         
    num_train_epochs=2,

    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    logging_steps=10,
    save_steps=500,
    report_to=[],

    fp16=False,                   
    bf16=False,

    remove_unused_columns=False, 
)

print("==== DPO Config ====")
print("beta:", dpo_config.beta)
print("lr:", dpo_config.learning_rate)
print("epochs:", dpo_config.num_train_epochs)
print("====================")

# ======================================================
# 7. 初始化 DPOTrainer
# ======================================================
trainer = DPOTrainer(
    model=policy_model,
    ref_model=None,               # TRL 自动冻结初始 policy
    args=dpo_config,
    train_dataset=dataset,
    processing_class=tokenizer,
)

# ======================================================
# 8. 开始训练
# ======================================================
print("\n🚀 Start DPO training...")
trainer.train()

# ======================================================
# 9. 保存 DPO adapter
# ======================================================
trainer.save_model(OUTPUT_DIR)

print(f"\nDPO 训练完成，Adapter 已保存至：{OUTPUT_DIR}")
