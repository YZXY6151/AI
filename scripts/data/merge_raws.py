#!/usr/bin/env python3
import os

def merge_files(primary_path: str, cc_path: str, out_path: str):
    """
    合并 primary 和 cc 两份纯文本文件，去重后写入 out_path。
    """
    seen = set()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fout:
        for path in (primary_path, cc_path):
            with open(path, "r", encoding="utf-8") as fin:
                for line in fin:
                    t = line.strip()
                    if not t or t in seen:
                        continue
                    seen.add(t)
                    fout.write(t + "\n")
    print(f"合并完成，共 {len(seen)} 条记录写入 {out_path}")

if __name__ == "__main__":
    # 默认文件路径，可根据需要调整
    primary = "data/raw.txt"
    cc      = "data/cc.txt"
    out     = "data/all_raw.txt"
    merge_files(primary, cc, out)
