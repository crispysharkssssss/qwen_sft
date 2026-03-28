"""
08_rag.py
混合 RAG 模式：
1. 读取外部 covid_news.txt
2. 如果检索到的内容包含答案 -> 优先基于检索内容回答。
3. 如果检索到的内容无关 -> 模型自动回退到自身知识库回答。
"""

import os
import torch
import csv
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer, util

# =========================
# 1. 命令行参数
# =========================
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="E:/models/Qwen2.5/Qwen/Qwen2___5-1___5B-Instruct", help="模型路径")
parser.add_argument("--kb_file", type=str, default="covid_news.txt", help="知识库文件")
parser.add_argument("--output_csv", type=str, default="rag_hybrid_result.csv", help="输出文件")
args = parser.parse_args()

# =========================
# 2. 读取文件
# =========================
if not os.path.exists(args.kb_file):
    raise FileNotFoundError(f"请确保目录下存在 {args.kb_file} 文件！")

print(f"🔹 读取知识库: {args.kb_file}")
with open(args.kb_file, "r", encoding="utf-8") as f:
    raw_text = f.read()

kb_sentences = [line.strip() for line in raw_text.split('\n') if len(line.strip()) > 10]

# =========================
# 3. 混合问题列表 (测试两种情况)
# =========================

# 混合列表：前4个答案在文件里，后4个答案不在文件里（需要模型自己答）
test_questions = [
    # --- 应该从文件中找答案 ---
    "2025年第一季度主要流行的毒株是什么？",
    "推荐接种的疫苗叫什么名字？",
    "居家隔离现在要求多少天？",
    "学校什么时候需要远程学习？",
    
    # --- 文件里没有，需要模型自己答 ---
    "请问番茄炒蛋怎么做？",
    "给我讲一个冷笑话。",
    "Python中列表和元组有什么区别？",
    "鲁迅原本姓什么？"
]

# =========================
# 4. 加载模型
# =========================
print("加载 Embedding 模型...")
embedder = SentenceTransformer("shibing624/text2vec-base-chinese")
kb_embeddings = embedder.encode(kb_sentences, convert_to_tensor=True)

print(f"加载 LLM: {args.model_path} ...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    device_map="auto"
)
model.eval()

# =========================
# 5. 智能检索函数 (核心修改)
# =========================
def get_hybrid_answer(question):
    # 1. 无论什么问题，先去库里找最相关的一段 
    query_emb = embedder.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(query_emb, kb_embeddings)
    top_idx = torch.argmax(scores)
    
    # 获取最相似的文本片段，不管它是否真的相关
    retrieved_context = kb_sentences[top_idx]
    
    # 2. 构建 "混合指令" Prompt
    # 关键在于告诉模型：参考资料仅供参考，不对就忽略
    prompt = (
        "### 指令：\n"
        "请回答用户的【问题】。\n"
        "为了辅助你，我检索了一段【参考资料】。\n"
        "规则如下：\n"
        "1. 如果【参考资料】中包含问题的答案，请**优先**基于资料回答。\n"
        "2. 如果【参考资料】与问题无关（例如问题是关于做菜，资料是关于疫情），请**忽略**资料，直接利用你自己的知识回答。\n\n"
        f"### 参考资料：\n{retrieved_context}\n\n"
        f"### 问题：\n{question}\n\n"
        "### 回答：\n"
    )

    # 3. 生成参数调整
    # temperature 稍微调高一点点 (0.1 -> 0.4)，让它在找不到资料时能流畅地“瞎编”通用知识
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.4, 
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(output_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    return response, retrieved_context

# =========================
# 6. 执行与保存
# =========================
print(f"\n>>> 开始混合推理，结果保存至 {args.output_csv} ...")

results = []
for q in tqdm(test_questions):
    ans, context_used = get_hybrid_answer(q)
    results.append({
        "Question": q, 
        "Model_Answer": ans,
        "Retrieved_Context_Used": context_used[:50] + "..." # 只记录前50字方便查看检索了啥
    })

with open(args.output_csv, "w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["Question", "Model_Answer", "Retrieved_Context_Used"])
    writer.writeheader()
    writer.writerows(results)

print("\n推理完成！")
# 简单打印结果预览
for item in results:
    print(f"Q: {item['Question']}\nA: {item['Model_Answer']}\n---")