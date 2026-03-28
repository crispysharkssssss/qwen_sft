"""
08_compare_results.py

对比 Base 模型 vs SFT+DPO 模型的 benchmark CSV 结果
用于分析微调与对齐带来的提升效果
"""

import csv
import statistics

BASE_CSV = "./results/benchmark_base.csv"
SFT_DPO_CSV = "./results/benchmark_sft_dpo.csv"

# =========================
# 1. 读取 CSV
# =========================
def load_csv(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    return rows

base_rows = load_csv(BASE_CSV)
dpo_rows = load_csv(SFT_DPO_CSV)

assert len(base_rows) == len(dpo_rows), "CSV 行数不一致，无法对比"

print(f"Loaded {len(base_rows)} examples for comparison")

# =========================
# 2. 数值指标对比
# =========================
def collect_scores(rows, key="Simple Score"):
    scores = []
    for r in rows:
        try:
            scores.append(float(r[key]))
        except:
            pass
    return scores

base_scores = collect_scores(base_rows)
dpo_scores = collect_scores(dpo_rows)

print("\n================ 数值对比 ================")
print(f"Base   Avg Score : {statistics.mean(base_scores):.4f}")
print(f"SFT+DPO Avg Score: {statistics.mean(dpo_scores):.4f}")
print(f"Δ Improvement   : {statistics.mean(dpo_scores) - statistics.mean(base_scores):.4f}")

# =========================
# 3. 不确定性 / 幻觉指标（如果存在）
# =========================
def count_true(rows, key):
    return sum(1 for r in rows if r.get(key, "").lower() == "true")

if "Has Uncertainty" in base_rows[0]:
    base_unc = count_true(base_rows, "Has Uncertainty")
    dpo_unc = count_true(dpo_rows, "Has Uncertainty")

    print("\n========== 不确定性表达（越多越好） ==========")
    print(f"Base   : {base_unc}")
    print(f"SFT+DPO: {dpo_unc}")
    print(f"Δ      : {dpo_unc - base_unc}")

if "Hallucination Risk" in base_rows[0]:
    base_hal = count_true(base_rows, "Hallucination Risk")
    dpo_hal = count_true(dpo_rows, "Hallucination Risk")

    print("\n========== 幻觉风险（越少越好） ==========")
    print(f"Base   : {base_hal}")
    print(f"SFT+DPO: {dpo_hal}")
    print(f"Δ      : {base_hal - dpo_hal}")

# =========================
# 4. 挑选“明显提升”的样例
# =========================
print("\n================ 典型改进样例（Top 5） ================")

improvements = []

for b, d in zip(base_rows, dpo_rows):
    try:
        delta = float(d["Simple Score"]) - float(b["Simple Score"])
    except:
        continue

    improvements.append({
        "question": b["Question"],
        "base_ans": b["Model Output"],
        "dpo_ans": d["Model Output"],
        "delta": delta,
    })

# 按提升幅度排序
improvements.sort(key=lambda x: x["delta"], reverse=True)

for i, ex in enumerate(improvements[:5], 1):
    print(f"\nCase {i} | Score +{ex['delta']:.3f}")
    print("Q:", ex["question"])
    print("\n[Base Answer]")
    print(ex["base_ans"])
    print("\n[SFT+DPO Answer]")
    print(ex["dpo_ans"])

print("\nComparison finished.")
