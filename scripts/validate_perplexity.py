#!/usr/bin/env python3
import math
import csv
from transformers import AutoTokenizer, AutoModelForCausalLM
import evaluate
import torch

# 1. 加载模型与 tokenizer（你也可换成 GPT-2 等更轻量模型）
MODEL = "models/yi-1.5-9b"
tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
model     = AutoModelForCausalLM.from_pretrained(MODEL, local_files_only=True).cuda().eval()

# 2. 加载 Hugging Face Evaluate 的 perplexity metric
ppl_metric = evaluate.load("perplexity", module_type="metric")

# 3. 读取样本（随机抽样 1k）
import random
with open("data/pretrain.txt", encoding="utf-8") as f:
    lines = [l.strip() for l in f if l.strip()]
sample = random.sample(lines, k=1000)

# 4. 计算 perplexity
results = []
for sent in sample:
    # 4.1 tokenizer
    enc = tokenizer(sent, return_tensors="pt").to("cuda")
    # 4.2 forward + loss
    with torch.no_grad():
        out = model(**enc, labels=enc["input_ids"])
    loss = out.loss.item()
    ppl  = math.exp(loss)
    results.append((sent, ppl))

# 5. 排序输出最流畅&最可疑
results.sort(key=lambda x: x[1])
with open("debug/ppl_hf.tsv", "w", encoding="utf-8") as f:
    writer = csv.writer(f, delimiter="\t")
    writer.writerow(["sentence", "ppl"])
    for sent, ppl in results:
        writer.writerow([sent, f"{ppl:.2f}"])

print("最低 PPL（最流畅）：")
for sent, ppl in results[:10]:
    print(f"{ppl:.2f}\t{sent}")

print("\n最高 PPL（疑似脏句）：")
for sent, ppl in results[-10:]:
    print(f"{ppl:.2f}\t{sent}")
