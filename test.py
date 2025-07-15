from datasets import load_dataset
import os

# 确保输出目录存在
os.makedirs("data/dialogs_raw", exist_ok=True)

# 加载 GoEmotions 数据集
goemotions = load_dataset("go_emotions")

# 保存训练集为 JSONL
goemotions["train"].to_json("data/dialogs_raw/go_emotions.jsonl")

print("✅ 已保存 go_emotions 至 dialogs_raw/")
