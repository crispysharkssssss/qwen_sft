import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

# =========================
# 1. 强制检查 CUDA
# =========================
assert torch.cuda.is_available(), "CUDA 不可用，请确认已安装 CUDA 版 PyTorch"

print("CUDA 可用")
print("GPU:", torch.cuda.get_device_name(0))
print("PyTorch CUDA:", torch.version.cuda)

torch.cuda.reset_peak_memory_stats()

# =========================
# 2. 指定本地模型路径（不下载）
# =========================
MODEL_DIR = r"E:\models\Qwen2.5\Qwen\Qwen2___5-1___5B-Instruct"

print("\n使用本地模型路径：")
print(MODEL_DIR)

# =========================
# 3. 加载 Tokenizer
# =========================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_DIR,
    use_fast=True,
    trust_remote_code=True
)

# =========================
# 4. 4-bit 量化加载模型
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)

print("\n正在以 4-bit 量化方式加载模型...")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    trust_remote_code=True
)

model.eval()

# =========================
# 5. 推理测试
# =========================
prompt = "你好，请介绍一下你自己"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

response = tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
)

print("\n模型输出：")
print(response)

# =========================
# 6. 显存统计
# =========================
torch.cuda.synchronize()
peak_mem = torch.cuda.max_memory_allocated() / 1024**2
print(f"\nCUDA 峰值显存占用：{peak_mem:.2f} MB")
