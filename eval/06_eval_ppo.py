import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from trl import AutoModelForCausalLMWithValueHead

# =========================
# 0. 配置
# =========================
BASE_MODEL = "E:/models/Qwen2.5/Qwen/Qwen2___5-1___5B-Instruct"
PPO_ADAPTER_DIR = "./output/ppo_adapter_v2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =========================
# 1. 加载 tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================
# 2. 加载 base + PPO Adapter + ValueHead
# =========================
print("🔹 Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
base_model.config.use_cache = True

print("🔹 Loading PPO Adapter...")
peft_model = PeftModel.from_pretrained(base_model, PPO_ADAPTER_DIR, is_trainable=False)
model = AutoModelForCausalLMWithValueHead(peft_model)
model.eval()
model.to(DEVICE)

# =========================
# 3. 测试 prompt 列表
# =========================
prompts = [
    "我最近持续头痛并伴有恶心，可能是什么原因？",
    "我父亲有高血压，最近血压控制不好，应该注意什么？",
    "孩子反复发烧三天了，但精神还可以，需要马上去医院吗？",
    "我长期失眠、心慌，是不是神经衰弱？",
    "胃痛伴随黑便，这种情况严重吗？",
]

# =========================
# 4. 奖励函数（同 PPO 训练）
# =========================
def medical_reward_fn(texts):
    rewards = []
    for t in texts:
        r = 0.0
        if "建议" in t: r += 0.5
        if "请" in t: r += 0.2
        if len(t) < 20: r -= 1.0
        rewards.append(r)
    return rewards

# =========================
# 5. 生成 + 评估
# =========================
generation_kwargs = {
    "max_new_tokens": 128,
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.7,
    "pad_token_id": tokenizer.pad_token_id,
    "eos_token_id": tokenizer.eos_token_id
}

print("\n inference + evaluation...\n")

all_rewards = []
for i, prompt in enumerate(prompts):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    
    # 去掉 prompt 部分
    response_ids = outputs[:, inputs.input_ids.shape[1]:]
    response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
    
    # 计算奖励
    reward = medical_reward_fn([response_text])[0]
    all_rewards.append(reward)
    
    print(f"[Prompt {i+1}]")
    print(f" Input: {prompt}")
    print(f" Response: {response_text}")
    print(f" Reward: {reward:.2f}\n")

# =========================
# 6. 平均奖励
# =========================
avg_reward = sum(all_rewards) / len(all_rewards)
print(f"Average reward over {len(prompts)} prompts: {avg_reward:.2f}")
