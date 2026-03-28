"""
05_train_ppo.py
基于 SFT Adapter 的 PPO 微调（TRL 0.8.6）
保留原始问候前缀 + 输出医疗建议
KL penalty + 问候奖励
"""

import os
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

# =========================
# 0. 基础环境
# =========================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
BASE_MODEL = "E:/models/Qwen2.5/Qwen/Qwen2___5-1___5B-Instruct"
SFT_ADAPTER = "./output/sft_adapter"
DATA_PATH = "./data/edical_dpo_data.json"
OUTPUT_DIR = "./output/ppo_adapter_v2"

# =========================
# 1. 4-bit 配置
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
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================
# 3. 加载 base + Adapter + ValueHead
# =========================
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL, quantization_config=bnb_config, torch_dtype=torch.float16, device_map="auto"
)
base_model.config.use_cache = False

print("Loading SFT Adapter...")
peft_model = PeftModel.from_pretrained(base_model, SFT_ADAPTER, is_trainable=True)

print("Wrapping model with ValueHead...")
model = AutoModelForCausalLMWithValueHead(peft_model)

# 修复 TRL 0.8.6 需要的属性
model.is_peft_model = True
if not hasattr(model, "is_sequential_parallel"):
    model.is_sequential_parallel = False

# 确保 v_head 可训练
for name, param in model.named_parameters():
    if "v_head" in name:
        param.requires_grad = True

# =========================
# 4. PPOConfig（加入 KL 约束）
# =========================
ppo_config = PPOConfig(
    batch_size=1,
    learning_rate=5e-6,
    mini_batch_size=1,
    gradient_accumulation_steps=1
)


# =========================
# 5. Dataset
# =========================
raw_dataset = load_dataset("json", data_files=DATA_PATH, split="train")

def build_ppo_dataset(example):
    if "prompt" not in example or not example["prompt"].strip():
        return None
    return {"input_ids": tokenizer(example["prompt"], truncation=True, max_length=512, padding=False)["input_ids"]}

train_dataset = raw_dataset.map(build_ppo_dataset, remove_columns=raw_dataset.column_names)
train_dataset = train_dataset.filter(lambda x: x is not None)
train_dataset.set_format(type="torch")

# =========================
# 6. 初始化 PPOTrainer
# =========================
# ref_model = 同样的 base_model + adapter（非训练模式）
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ref_model = AutoModelForCausalLMWithValueHead(
    PeftModel.from_pretrained(base_model, SFT_ADAPTER)
)
ref_model.eval()
ref_model.to(device) 
ppo_trainer = PPOTrainer(
    model=model,
    ref_model=ref_model,
    config=ppo_config,
    tokenizer=tokenizer,
    dataset=train_dataset
)

# 在 step 时传入 kl_target / kl_coef
kl_coef = 0.2  # 保留原始 SFT 风格

# =========================
# 7. 自定义奖励函数
# =========================
warm_prefixes = [
    "很抱歉听到您的情况，请仔细阅读以下建议，每一点都非常重要：",
    "了解到您的不适，我非常难过。根据您的描述，我将分条说明注意事项：",
    "专业提示：请严格参考下面的步骤进行自我管理：",
    "非常抱歉听到您的问题，医学指导如下，请逐条遵守：",
    "针对您的症状，我整理了详细的分析和建议：",
    "下面的内容是医生风格解答，请认真阅读：",
    "我将以医生专业视角，逐条说明可能原因和处理方法："
]
def medical_reward_fn(texts):
    rewards = []
    for t in texts:
        r = 0.0
        # 奖励问候前缀
        if any(prefix in t for prefix in warm_prefixes):
            r += 0.5
        # 太短惩罚
        if len(t) < 20: r -= 1.0
        rewards.append(r)
    return rewards

# =========================
# 8. 训练循环
# =========================
def find_generating_model(model):
    """递归查找可 generate 的模型"""
    if hasattr(model, "generate") and callable(model.generate):
        return model
    for attr in ["pretrained_model", "module", "policy", "model", "_orig_mod"]:
        if hasattr(model, attr) and getattr(model, attr) is not model:
            found = find_generating_model(getattr(model, attr))
            if found: return found
    return None

generation_kwargs = {
    "max_new_tokens": 128,
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.7,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id
}

num_steps = 200  # 推荐训练步数
current_device = ppo_trainer.accelerator.device

print("\nStart PPO training with KL penalty + greeting reward...\n")

for step, batch in enumerate(ppo_trainer.dataloader):
    query_ids = batch["input_ids"].to(current_device)

    with torch.no_grad():
        gen_model = find_generating_model(ppo_trainer.model)
        if gen_model is None:
            gen_model = model.pretrained_model
        gen_model.eval()
        response_ids_full = gen_model.generate(query_ids, **generation_kwargs)
        gen_model.train()

    # 切掉 query 部分
    response_ids = response_ids_full[:, query_ids.shape[1]:]
    response_texts = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

    # 计算奖励
    raw_rewards = medical_reward_fn(response_texts)
    rewards = [torch.tensor(r, dtype=torch.float32, device=current_device) for r in raw_rewards]

    # 将 Tensor -> List[Tensor] 传入 step
    query_list = [q for q in query_ids]
    response_list = [r for r in response_ids]

    stats = ppo_trainer.step(query_list, response_list, rewards)

    if step % 10 == 0:
        print(f"[Step {step}] reward={raw_rewards[0]:.2f} | len={len(response_texts[0])}")

    if step >= num_steps:
        break

# =========================
# 9. 保存 Adapter
# =========================
print(f"\nSaving adapter to {OUTPUT_DIR}")
ppo_trainer.save_pretrained(OUTPUT_DIR)
print("Done!")
