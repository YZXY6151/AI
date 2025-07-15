#!/usr/bin/env python3
import argparse
import sys
import multiprocessing as mp
from datasketch import MinHash, MinHashLSH
from tqdm import tqdm

def get_minhash(text: str, num_perm: int, k_shingle: int = 5) -> MinHash:
    shingles = (text[i:i + k_shingle] for i in range(len(text) - k_shingle + 1))
    m = MinHash(num_perm=num_perm)
    for sh in shingles:
        m.update(sh.encode('utf8'))
    return m

def worker(args):
    idx, text, num_perm, k_shingle = args
    return idx, text, get_minhash(text, num_perm, k_shingle)

def main():
    parser = argparse.ArgumentParser(description="Fast MinHash-LSH deduplication")
    parser.add_argument("-i", "--input", required=True,
                        help="Input file path or '-' for stdin")
    parser.add_argument("-o", "--output", required=True,
                        help="Path to write deduplicated output")
    parser.add_argument("-t", "--threshold", type=float, default=0.8,
                        help="MinHash similarity threshold (0.0–1.0)")
    parser.add_argument("-n", "--num_perm", type=int, default=128,
                        help="Number of MinHash permutations")
    parser.add_argument("-p", "--processes", type=int, default=mp.cpu_count(),
                        help="Number of parallel processes")
    parser.add_argument("-k", "--shingle_size", type=int, default=5,
                        help="Shingle size (in characters)")
    args = parser.parse_args()

    # 1. 读取所有输入
    if args.input == "-":
        lines = [line.strip() for line in sys.stdin if line.strip()]
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

    print(f"读取样本总数: {len(lines)} 行")

    # 2. 并行生成 MinHash
    job_args = [(i, line, args.num_perm, args.shingle_size) for i, line in enumerate(lines)]
    with mp.Pool(args.processes) as pool:
        results = list(tqdm(pool.imap(worker, job_args, chunksize=500),
                            total=len(job_args), desc="生成 MinHash", mininterval=60))

    # 3. 插入到 LSH 并去重
    lsh = MinHashLSH(threshold=args.threshold, num_perm=args.num_perm)
    unique_lines = []
    for idx, text, m in tqdm(results, desc="去重中", mininterval=60):
        if not lsh.query(m):
            lsh.insert(str(idx), m)
            unique_lines.append(text)

    # 4. 写入结果
    with open(args.output, "w", encoding="utf-8") as f:
        for line in unique_lines:
            f.write(line + "\n")

    print(f"✅ 去重完成：{len(lines)} → {len(unique_lines)} 行")

if __name__ == "__main__":
    main()
