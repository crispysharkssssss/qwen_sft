"""
01_data_prepare.py

功能：
- 从 ChatMed_Consult_Dataset 加载数据
- 数据清洗（过滤短回答） + 数据增强（医生风格化前缀）
- 转换为 Alpaca 格式 JSON 保存
"""

import json
import random
from datasets import load_dataset

# =========================
# 1. 医生风格化前缀（数据增强）
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

# =========================
# 2. 加载数据集
# =========================
try:
    print("📥 尝试加载 michaelwzhu/ChatMed_Consult_Dataset ...")
    dataset = load_dataset(
        "michaelwzhu/ChatMed_Consult_Dataset",
        split="train"
    )
    source = "michaelwzhu/ChatMed_Consult_Dataset"
except Exception as e:
    print("主数据集加载失败，尝试使用备选数据集 shibing624/medical")
    print("错误信息：", e)
    dataset = load_dataset(
        "shibing624/medical",
        split="train"
    )
    source = "shibing624/medical"

print(f"成功加载数据集：{source}")

# =========================
# 3. 记录原始样本数并截取前 3000 条
# =========================
raw_count = len(dataset)
dataset = dataset.select(range(min(3000, raw_count)))

# =========================
# 4. 数据清洗 + 增强 + 转 Alpaca 格式
# =========================
alpaca_data = []

for sample in dataset:
    # 兼容字段
    instruction = sample.get("question") or sample.get("query") or sample.get("prompt")
    response = sample.get("response") or sample.get("answer") or sample.get("reply")

    if not instruction or not response:
        continue

    # 过滤短回答（少于 15 个字符）
    if len(response.strip()) < 15:
        continue

    # 随机插入医生风格化前缀 + 换行
    prefix = random.choice(warm_prefixes)
    enhanced_response = prefix + "\n" + response.strip()

    alpaca_data.append({
        "instruction": instruction.strip(),
        "input": "",
        "output": enhanced_response
    })

clean_count = len(alpaca_data)

# =========================
# 5. 保存为 JSON
# =========================
output_file = "medical_sft_data.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(alpaca_data, f, ensure_ascii=False, indent=2)

# =========================
# 6. 输出统计与样例
# =========================
print("\n数据统计：")
print(f"清洗前样本数：{raw_count}")
print(f"清洗后样本数：{clean_count}")

if alpaca_data:
    print("\n🔍 样例数据：")
    print(json.dumps(alpaca_data[0], ensure_ascii=False, indent=2))

print(f"\n数据已保存至：{output_file}")
