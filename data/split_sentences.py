#!/usr/bin/env python3
import argparse
import os
from multiprocessing import Pool, cpu_count
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
from tqdm import tqdm

# Global state for worker processes
TOKENIZER = None
MAX_TOKENS = None

def init_worker(model_path, max_tokens):
    """Worker initializer: load tokenizer and set max_tokens."""
    global TOKENIZER, MAX_TOKENS
    TOKENIZER = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    MAX_TOKENS = max_tokens

def process_line(line):
    """
    Split a line into sentences, filter by token length,
    return list of valid fragments.
    """
    fragments = []
    for sent in sent_tokenize(line.strip()):
        token_ids = TOKENIZER.encode(sent, add_special_tokens=False)
        if len(token_ids) <= MAX_TOKENS:
            fragments.append(sent)
    return fragments

def split_and_filter(input_path, output_path, max_tokens, model_path, workers):
    """Main function: parallel split & filter."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_lines = sum(1 for _ in open(input_path, 'r', encoding='utf-8', errors='ignore'))
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout, \
         Pool(processes=workers, initializer=init_worker, initargs=(model_path, max_tokens)) as pool:

        for fragments in tqdm(pool.imap(process_line, fin, chunksize=1000), total=total_lines, desc="Splitting"):
            for frag in fragments:
                fout.write(frag + "\n")

def parse_args():
    parser = argparse.ArgumentParser(description="多进程分句与切片脚本")
    parser.add_argument("-i", "--input", required=True, help="输入去重文本文件路径")
    parser.add_argument("-o", "--output", required=True, help="输出预训练文本文件路径")
    parser.add_argument("-m", "--max_tokens", type=int, default=512, help="最大 token 数，默认 512")
    parser.add_argument("-p", "--model_path", default="models/yi-1.5-9b", help="模型目录，用于 tokenizer")
    parser.add_argument("-w", "--workers", type=int, default=cpu_count(), help="并行进程数，默认全核")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    split_and_filter(
        input_path=args.input,
        output_path=args.output,
        max_tokens=args.max_tokens,
        model_path=args.model_path,
        workers=args.workers
    )
