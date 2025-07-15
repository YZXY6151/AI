#!/usr/bin/env python3
import json
import random
import hashlib
import os
import glob
from collections import Counter
import statistics

# 1. 配置
random.seed(42)
raw_dir     = "data/dialogs_raw"
output_dir  = "data/finetune"
valid_frac  = 0.1  # 验证集比例

# 2. 确保输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 3. 遍历所有文件，并自动识别数据源
examples = []
input_files = glob.glob(os.path.join(raw_dir, "*.jsonl"))
for path in input_files:
    source = os.path.basename(path).replace(".jsonl", "")
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            ex = {"metadata": {"source": source}}

            # 识别三种格式
            if source == "personachat":
                prompt = (
                    "Persona: " + " ".join(obj.get("personality", [])) +
                    "\nHistory:\n" + "\n".join(obj.get("history", []))
                )
                completion = obj.get("candidates", [""])[0]
                ex["metadata"]["personality"] = obj.get("personality", [])
            elif source == "dailydialog":
                dialog = obj.get("dialog", [])
                prompt = "\n".join(dialog[:-1])
                completion = dialog[-1] if dialog else ""
                ex["metadata"].update({
                    "act":     obj.get("act", [])[-1]     if obj.get("act")     else None,
                    "emotion": obj.get("emotion", [])[-1] if obj.get("emotion") else None
                })
            elif source == "go_emotions":
                prompt = obj.get("text", "")
                completion = " ".join(f"<label_{lbl}>" for lbl in obj.get("labels", []))
                ex["metadata"]["labels"] = obj.get("labels", [])
            else:
                continue  # 忽略未知源

            ex["prompt"] = prompt
            ex["completion"] = completion

            h = hashlib.md5((prompt + completion).encode("utf-8")).hexdigest()
            ex["metadata"]["hash"] = h
            examples.append(ex)

# 4. 去重
unique = {}
for ex in examples:
    h = ex["metadata"]["hash"]
    if h not in unique:
        unique[h] = ex
examples = list(unique.values())

# 5. 打乱并切分
random.shuffle(examples)
n_total = len(examples)
n_valid = int(n_total * valid_frac)
train = examples[n_valid:]
valid = examples[:n_valid]

# 6. 保存
with open(os.path.join(output_dir, "train.jsonl"), 'w', encoding='utf-8') as f:
    for ex in train:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")
with open(os.path.join(output_dir, "valid.jsonl"), 'w', encoding='utf-8') as f:
    for ex in valid:
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

# 7. 分布信息
labels_counter = Counter()
for ex in examples:
    if ex["metadata"]["source"] == "go_emotions":
        for lbl in ex["metadata"].get("labels", []):
            labels_counter[lbl] += 1

lengths = [len(ex["prompt"].split()) for ex in examples]

print(f"✅ 总样本数: {n_total}")
print(f"✅ Train / Valid: {len(train)} / {len(valid)}")
print("✅ GoEmotions 标签分布 (前 10):", dict(labels_counter.most_common()[:10]), "…")
print(f"✅ Prompt 长度 (词数估算) 均值: {statistics.mean(lengths):.2f}, 标准差: {statistics.pstdev(lengths):.2f}")
