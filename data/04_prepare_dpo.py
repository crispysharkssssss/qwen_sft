"""
04_prepare_dpo.py

基于 SFT 数据，构造「温和但有效」的 DPO 偏好数据
特点：
- rejected 仍然是“医生回答”
- 但缺少：同理心 / 结构 / 风险提醒
- 不引入明显医学错误
"""

import json
import random

INPUT_FILE = "medical_sft_data.json"
OUTPUT_FILE = "medical_dpo_data.json"

# =========================
# 1. 几种模板
# =========================

def weaken_empathy(text: str) -> str:
    """去掉医生情绪与关怀语句"""
    lines = text.splitlines()
    lines = [
        l for l in lines
        if not any(k in l for k in ["抱歉", "理解", "难过", "请您", "非常"])
    ]
    return "\n".join(lines).strip()

def weaken_structure(text: str) -> str:
    """合并分点，降低条理性"""
    text = text.replace("：\n", "：")
    text = text.replace("\n1.", " ")
    text = text.replace("\n2.", " ")
    text = text.replace("\n3.", " ")
    text = text.replace("\n", " ")
    return text.strip()

def weaken_safety(text: str) -> str:
    """去掉部分风险与就医提示（但不完全移除）"""
    lines = text.splitlines()
    lines = [
        l for l in lines
        if not any(k in l for k in ["尽快就医", "建议检查", "需要排查", "严重"])
    ]
    return "\n".join(lines).strip()

WEAKEN_FUNCS = [
    weaken_empathy,
    weaken_structure,
    weaken_safety,
]

# =========================
# 2. 加载 SFT 数据
# =========================

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    sft_data = json.load(f)

dpo_data = []

# =========================
# 3. 构造 DPO 样本
# =========================

for sample in sft_data:
    instruction = sample["instruction"].strip()
    input_text = sample.get("input", "").strip()
    chosen = sample["output"].strip()

    prompt = instruction if not input_text else instruction + "\n" + input_text

    # 随机选择 1–2 种轻度劣化方式
    rejected = chosen
    for fn in random.sample(WEAKEN_FUNCS, k=random.choice([1, 2])):
        rejected = fn(rejected)

    # 兜底：防止 rejected 太短或被削没了
    if len(rejected) < 30:
        rejected = "根据您的描述，可能与多种因素有关，建议结合自身情况观察，如有不适可咨询医生。"

    dpo_data.append({
        "prompt": prompt,
        "chosen": chosen,
        "rejected": rejected
    })

# =========================
# 4. 保存
# =========================

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    json.dump(dpo_data, f, ensure_ascii=False, indent=2)

print(f"DPO 数据构造完成：{OUTPUT_FILE}")
print(f"样本数：{len(dpo_data)}")

# 打印一个示例
print("\n示例：")
print(json.dumps(dpo_data[0], ensure_ascii=False, indent=2))
