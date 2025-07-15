#!/usr/bin/env python3
# File: scripts/eval_ppl.py

import argparse
import math
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

def batchify(lines, batch_size):
    batch = []
    for ln in lines:
        batch.append(ln)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def compute_ppl(input_path, model_dir, batch_size, max_length):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 本地加载
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_dir), local_files_only=True)
    model.to(device).eval()

    total_loss = 0.0
    total_tokens = 0

    with open(input_path, "r", encoding="utf-8") as f:
        lines = (ln.strip() for ln in f if ln.strip())

        for batch in tqdm(batchify(lines, batch_size), desc="Evaluating PPL"):
            enc = tokenizer(batch,
                            return_tensors="pt",
                            padding="longest",
                            truncation=True,
                            max_length=max_length)
            input_ids = enc.input_ids.to(device)
            attn_mask = enc.attention_mask.to(device)

            with torch.no_grad():
                outputs = model(input_ids,
                                attention_mask=attn_mask,
                                labels=input_ids)
                # outputs.loss 是批内平均 NLL
                loss = outputs.loss.item()

            # 计算实际非 pad 的 token 数
            pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
            real_tokens = (input_ids != pad_id).sum().item()

            total_loss += loss * real_tokens
            total_tokens += real_tokens

    avg_nll = total_loss / total_tokens
    ppl = math.exp(avg_nll)
    print("\n>>> 总 token 数:", total_tokens)
    print(f">>> 平均 NLL: {avg_nll:.4f}")
    print(f">>> Perplexity: {ppl:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="按批算困惑度")
    parser.add_argument("-i", "--input", required=True,
                        help="输入文本，每行一句（推荐先抽样）")
    parser.add_argument("-m", "--model", required=True,
                        help="本地模型目录")
    parser.add_argument("-b", "--batch_size", type=int, default=8,
                        help="批大小，显存／RAM 受限可调小")
    parser.add_argument("-l", "--max_length", type=int, default=512,
                        help="最大 token 长度，超长截断")
    args = parser.parse_args()

    model_dir = Path(args.model)
    if not model_dir.exists():
        parser.error(f"模型目录不存在：{model_dir}")

    compute_ppl(args.input, model_dir, args.batch_size, args.max_length)
