import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
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
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

# =========================
# 加载基座模型
# =========================
print("🔹 加载基座模型 (4-bit)...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True
)
base_model.eval()

# =========================
# 加载 SFT Adapter
# =========================
print(f"🔹 加载 SFT Adapter: {ADAPTER_PATH}")
sft_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
sft_model.eval()

# =========================
# 推理函数
# =========================
def generate_answer(
    model,
    prompt,
    max_new_tokens=256,
    temperature=0.7,
    greedy=False
):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if not greedy else 0.0,
            do_sample=not greedy,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
        )
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return answer

# =========================
# 交互式评测
# =========================
print("\n进入交互式评测模式")
print("输入问题后，系统将同时输出：")
print("  - 基座模型回答")
print("  - SFT 微调模型回答")
print("输入 exit / quit / q 退出\n")

while True:
    try:
        instruction = input("请输入测试问题 >>> ").strip()
        if instruction.lower() in ["exit", "quit", "q"]:
            print("退出评测")
            break

        if not instruction:
            continue

        print("\n基座模型生成中...")
        base_answer = generate_answer(base_model, instruction)

        print("SFT 模型生成中...")
        sft_answer = generate_answer(sft_model, instruction)

        print("\n==================== 对比结果 ====================")
        print(f"问题:\n{instruction}")
        print("\n[Base 模型回答]:")
        print(base_answer)
        print("\n[SFT 模型回答]:")
        print(sft_answer)
        print("=================================================\n")

    except KeyboardInterrupt:
        print("\n手动中断，退出评测")
        break
