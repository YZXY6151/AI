#!/usr/bin/env python3
import re
import sys

def is_title(line):
    # 全大写或首字母大写、连续多个词且无标点 → 目录式标题
    return re.match(r'^[A-Z][A-Za-z0-9]+(?: [A-Z][A-Za-z0-9]+){2,}$', line)

def has_code(line):
    # 常见代码关键字
    code_kw = ['import ', 'def ', 'function ', '<script', '#include', 'printf', 'console.log']
    return any(kw in line for kw in code_kw)

def has_contact(line):
    # 邮编或电话号码
    if re.search(r'\d{2,5}(?:-\d{2,5})?', line): return True
    if re.search(r'\d{3,4}[- ]?\d{7,8}', line): return True
    return False

def is_short_or_long(line, min_tok=5, max_tok=200):
    toks = line.strip().split()
    return len(toks) < min_tok or len(toks) > max_tok

def clean(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8', errors='ignore') as fin, \
         open(output_path, 'w', encoding='utf-8') as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue

            # —— 新增：以数字开头的行全部跳过 ——
            if re.match(r'^[0-9]', line):
                continue

            # 1. 标题/目录式标题
            if is_title(line):
                continue

            # 2. 代码片段
            if has_code(line):
                continue

            # 3. 广告/地名/联系方式
            if has_contact(line):
                continue

            # 5. 过长或过短
            if is_short_or_long(line):
                continue

            # 6. 日志/反爬虫痕迹
            lower = line.lower()
            if 'detected behavior' in lower or 'user-agent' in lower:
                continue

            # 其他规则通过，保留
            fout.write(line + '\n')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print("Usage: python data/clean_pretrain.py <input> <output>")
        sys.exit(1)
    clean(sys.argv[1], sys.argv[2])
