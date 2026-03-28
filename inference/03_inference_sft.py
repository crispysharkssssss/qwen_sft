import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# =========================
# 配置
# =========================
BASE_MODEL = "E:/models/Qwen2.5/Qwen/Qwen2___5-1___5B-Instruct"
ADAPTER_PATH = "./output/sft_adapter"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# BitsAndBytes 4-bit 配置
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

# =========================
# 加载 Tokenizer
# =========================
print("🔹 加载 Tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# =========================
# 加载基座模型 (4-bit)
# =========================
print("🔹 加载基座模型 (4-bit)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
base_model_eval = base_model.eval()
base_model.config.use_cache = False
print("基座模型加载完成")

# =========================
# 加载 SFT Adapter
# =========================
print(f"🔹 加载 SFT Adapter: {ADAPTER_PATH}")
sft_model_eval = copy.deepcopy(base_model).eval()
sft_model_eval = PeftModel.from_pretrained(sft_model_eval, ADAPTER_PATH)
sft_model_eval.eval()
sft_model_eval.config.use_cache = False
print("SFT Adapter 加载完成 (基座模型保持不变)")
print("SFT Adapter 加载完成")

# =========================
# 推理函数
# =========================
def generate_answer(model, prompt, max_new_tokens=256, temperature=0.7):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=False,
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# =========================
# 交互式循环
# =========================
print("\n请输入你的问题 (输入 'exit' 或 'quit' 退出)：\n")

while True:
    user_input = input("用户: ").strip()
    if user_input.lower() in ["exit", "quit"]:
        break

    # SFT 模型回答
    sft_answer = generate_answer(sft_model_eval, user_input)

    # 基座模型回答
    base_answer = generate_answer(base_model_eval, user_input)

    # 打印对比
    print("\n===================== 对比 =====================")
    print(f"用户输入: {user_input}")
    print(f"\n[基座模型回答]:\n{base_answer}")
    print(f"\n[SFT 微调后回答]:\n{sft_answer}")
    print("================================================\n")
