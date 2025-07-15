#!/usr/bin/env python3
import argparse
import fasttext
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm

def parse_args():
    p = argparse.ArgumentParser(description="并行化快速英语行过滤（FastText + 多进程）")
    p.add_argument("--input", "-i", required=True,
                   help="输入文件，每行一句待过滤文本")
    p.add_argument("--output", "-o", default="data/lang_en.txt",
                   help="输出文件，只保留英语行")
    p.add_argument("--model", "-m",
                   default="lid.176.ftz",
                   help="FastText 语言识别模型文件路径")
    p.add_argument("--workers", "-w", type=int, default=cpu_count(),
                   help="并发进程数，默认使用全部 CPU 核")
    p.add_argument("--ascii_thresh", type=float, default=0.8,
                   help="ASCII 字符比例阈值，先做预过滤，默认 0.8")
    return p.parse_args()

def looks_english_ascii(line, thresh):
    """快速预过滤：ASCII 字符比例不足即直接丢弃"""
    if not line:
        return False
    cnt = sum(1 for c in line if ord(c) < 128)
    return (cnt / len(line)) >= thresh

def init_model(model_path):
    global clf
    clf = fasttext.load_model(model_path)

def detect_en(line):
    """
    单行判定：先 ASCII 预过滤，再 FastText 判断
    返回该行或者 None
    """
    line = line.strip()
    if not looks_english_ascii(line, ascii_thresh):
        return None

    # FastText 预测
    label, prob = clf.predict(line.replace("\n", " "), k=1)[0][0], clf.predict(line, k=1)[1][0]
    if label == "__label__en":
        return line
    return None

def worker(lines):
    """给多进程 Pool 用的简单包装"""
    out = []
    for line in lines:
        res = detect_en(line)
        if res:
            out.append(res)
    return out

def chunked_iterable(iterable, chunk_size):
    """把可迭代对象切块，返回一堆小列表"""
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(chunk_size):
                chunk.append(next(it))
        except StopIteration:
            if chunk:
                yield chunk
            break
        yield chunk

if __name__ == "__main__":
    args = parse_args()
    ascii_thresh = args.ascii_thresh

    if not os.path.exists(args.model):
        print(f"FastText 模型 {args.model} 不存在，请先下载：")
        print("  wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz")
        exit(1)

    # 预载模型到每个子进程
    pool = Pool(processes=args.workers, initializer=init_model, initargs=(args.model,))

    with open(args.input, "r", encoding="utf-8", errors="ignore") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        # 把大文件分片，每片 2000 行发给一个 worker
        total = sum(1 for _ in open(args.input, encoding="utf-8", errors="ignore"))
        pbar = tqdm(total=total, desc="Filtering English", unit=" lines")

        for chunk in chunked_iterable(fin, chunk_size=2000):
            # apply_async 比 imap 更灵活，但这里用 map 保序也 OK
            results = pool.apply(worker, (chunk,))
            for line in results:
                fout.write(line + "\n")
            pbar.update(len(chunk))

        pbar.close()
    pool.close()
    pool.join()
    print(f"Done! 英文行已写入 {args.output}")
