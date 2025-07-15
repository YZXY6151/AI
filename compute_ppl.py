import torch
import math
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "gpt2"
TEXT_PATH = "data/ppl_sample.txt"

# 加载 tokenizer 和 model（使用本地文件）
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True,
    local_files_only=True
)

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

@torch.no_grad()
def compute_perplexity(filepath, max_length=128):

    total_loss = 0.0
    total_tokens = 0

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # 编码并截断
            inputs = tokenizer(line, return_tensors="pt", truncation=True, max_length=max_length)
            input_ids = inputs["input_ids"].to(device)

            # 忽略过短的内容
            if input_ids.size(1) < 2:
                continue

            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss

            total_loss += loss.item() * input_ids.size(1)
            total_tokens += input_ids.size(1)

    if total_tokens == 0:
        return float("inf")
    return math.exp(total_loss / total_tokens)

if __name__ == "__main__":
    ppl = compute_perplexity(TEXT_PATH)
    print(f"\n📊 Perplexity on 1000 lines: {ppl:.2f}")
