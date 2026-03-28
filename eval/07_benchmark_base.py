"""
07_benchmark_base.py
TruthfulQA generation validation
Base Qwen 模型 benchmark
"""

import csv
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm import tqdm

# =========================
# 0. 路径配置
# =========================
BASE_MODEL = "E:/models/Qwen2.5/Qwen/Qwen2___5-1___5B-Instruct"
CSV_FILE = "./results/benchmark_base.csv"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# =========================
# 3. Base 模型
# =========================
print("Loading Base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
)
model.eval()

# =========================
# 4. 数据集
# =========================
dataset = load_dataset("truthful_qa", "generation", split="validation")
print(f"Dataset size: {len(dataset)}")

# =========================
# 5. CSV
# =========================
csv_f = open(CSV_FILE, "w", newline="", encoding="utf-8")
writer = csv.DictWriter(
    csv_f,
    fieldnames=["Question", "Best Answer", "Model Output", "Simple Score"]
)
writer.writeheader()

# =========================
# 6. 简单评分
# =========================
def simple_match_score(pred, target):
    pred_tokens = set(pred.split())
    target_tokens = set(target.split())
    if not pred_tokens:
        return 0.0
    return len(pred_tokens & target_tokens) / len(pred_tokens)

# =========================
# 7. 推理循环
# =========================
for ex in tqdm(dataset, desc="Base Benchmark"):
    question = ex["question"]
    best_answer = ex.get("best_answer", "")

    inputs = tokenizer(question, return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
        )

    answer_ids = outputs[:, inputs["input_ids"].shape[1]:]
    model_output = tokenizer.decode(answer_ids[0], skip_special_tokens=True)

    score = simple_match_score(model_output, best_answer)

    writer.writerow({
        "Question": question,
        "Best Answer": best_answer,
        "Model Output": model_output,
        "Simple Score": round(score, 3),
    })

csv_f.close()
print(f"Base benchmark finished → {CSV_FILE}")
