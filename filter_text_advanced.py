#!/usr/bin/env python3
import re

input_path  = 'data/final_dedup.txt'
output_path = 'data/final_dedup_text_only_v2.txt'

# 跳过“头部字段: 值”这种样式（如 WARC-Block-Digest: sha1:...）
header_pattern = re.compile(r'^[A-Za-z0-9-]+:')

# 过滤 HTML/JS/C 之类代码的通用标志
code_patterns = [
    re.compile(r'<[^>]+>'),                            # HTML 标签
    re.compile(r'\b(function|var|let|const|class)\b'), # JS 关键字
    re.compile(r'[{}();=+\[\]\/\\]'),                  # 常见代码符号
    re.compile(r'\b(import|export)\s'),                # 模块语法
    re.compile(r'//|/\*|\*/'),                         # 注释符
]

# 识别自然语言：至少两个“中文或英文单词” (长度 ≥4)
text_pattern = re.compile(r'\b[A-Za-z\u4e00-\u9fff]{4,}\b')

with open(input_path, 'r', encoding='utf-8', errors='ignore') as fin, \
     open(output_path, 'w', encoding='utf-8') as fout:
    for line in fin:
        raw = line.rstrip('\n')
        if not raw.strip():
            continue

        s = raw.lstrip()
        # 1) 跳过开头非字母/数字的行（如以 "!"、"-" 等开头）
        if not re.match(r'^[A-Za-z0-9\u4e00-\u9fff]', s):
            continue
        # 2) 跳过 header 样式
        if header_pattern.match(s):
            continue
        # 3) 跳过明显的代码行
        if any(p.search(s) for p in code_patterns):
            continue
        # 4) 只保留真正的自然语言行（至少两个单词）
        if len(text_pattern.findall(s)) >= 2:
            fout.write(s + '\n')

print(f"过滤完成 → {output_path}")
