#!/usr/bin/env python3
import time
import torch
import yaml
import argparse
from utils.model_loader import load_model  # 引用封装的加载接口

def load_config(path="config.yaml"):
    """
    从 config.yaml 中读取模型与推理配置。
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def benchmark(model, tokenizer, seq_len, batch_size, n_steps=20):
    """
    对模型进行推理基准测试。
    - seq_len: 序列长度
    - batch_size: 批量大小
    - n_steps: 重复步骤数（用于计算平均时间）
    """
    # 构造随机输入
    input_ids = torch.randint(
        0, tokenizer.vocab_size,
        (batch_size, seq_len),
        dtype=torch.long,
        device="cuda"
    )
    torch.cuda.empty_cache()

    # 预热
    with torch.no_grad():
        _ = model(input_ids).logits

    # 正式计时
    torch.cuda.synchronize()
    t0 = time.time()
    with torch.no_grad():
        for _ in range(n_steps):
            _ = model(input_ids).logits
    torch.cuda.synchronize()

    elapsed = time.time() - t0
    avg = elapsed / n_steps
    print(
        f"Mode: seq_len={seq_len}, batch_size={batch_size} -> "
        f"Total: {elapsed:.4f}s, Avg/step: {avg:.4f}s"
    )
    return elapsed, avg

def main():
    # 1. 加载配置
    cfg = load_config()
    model_path = cfg["model"]["path"]
    inference_cfg = cfg["inference"]

    # 2. 解析命令行参数
    parser = argparse.ArgumentParser(description="Benchmark Yi-1.5-9B-Chat model")
    parser.add_argument(
        "--mode", choices=inference_cfg.keys(), default="default",
        help="选择推理配置模式: " + ", ".join(inference_cfg.keys())
    )
    parser.add_argument(
        "--steps", "-s", type=int, default=20,
        help="迭代步数（默认 20）"
    )
    args = parser.parse_args()

    # 3. 根据模式读取推理参数
    mode_cfg = inference_cfg[args.mode]
    seq_len = mode_cfg["seq_len"]
    batch_size = mode_cfg["batch_size"]

    # 4. 加载模型与 tokenizer
    model, tokenizer = load_model(model_path)

    # 5. 执行基准测试
    benchmark(model, tokenizer, seq_len, batch_size, n_steps=args.steps)

if __name__ == "__main__":
    main()
