#!/usr/bin/env python3
import argparse
import hashlib
import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from warcio.archiveiterator import ArchiveIterator
import gzip
import os
import warnings
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def parse_args():
    parser = argparse.ArgumentParser(description="批量下载并清洗文本（支持 URL 列表或 WARC 文件）")
    parser.add_argument("--input", "-i", required=True,
                        help="输入路径：URL 列表文件（每行一个 URL）或 WARC 文件（.warc 或 .warc.gz）")
    parser.add_argument("--output", "-o", default="data/raw.txt",
                        help="输出文本文件路径，默认 data/raw.txt")
    parser.add_argument("--workers", "-w", type=int, default=4,
                        help="并发下载线程数（仅对 URL 列表生效），默认 4")
    parser.add_argument("--retries", "-r", type=int, default=3,
                        help="每个 URL 最大重试次数（仅对 URL 列表生效），默认 3")
    return parser.parse_args()

def iter_warc_records(warc_path):
    """
    迭代 WARC 文件里的每个 response 记录，Yield 原始 HTML bytes。
    """
    open_mode = 'rb'
    opener = gzip.open if warc_path.endswith(('.gz', '.gzip')) else open
    with opener(warc_path, open_mode) as stream:
        for record in ArchiveIterator(stream):
            if record.rec_type == 'response':
                html = record.content_stream().read()
                yield html

def clean_html(html) -> str:
    if isinstance(html, (bytes, bytearray)):
        html = html.decode("utf-8", errors="ignore")
    try:
        soup = BeautifulSoup(html, "lxml")  # 使用 lxml 解析器
    except Exception:
        return ""
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return " ".join(lines)


def read_urls(path: str) -> list[str]:
    """
    从文本文件读取 URL 列表（每行一个 URL），返回去重后的列表。
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = [ln.strip() for ln in f if ln.strip()]
    seen = set(); uniq = []
    for u in raw:
        if u not in seen:
            seen.add(u); uniq.append(u)
    return uniq

def download_one(url: str, retries: int = 3, timeout: int = 10) -> str:
    """
    下载单个 URL 并清洗 HTML，失败时重试 retries 次。
    返回纯文本或空字符串。
    """
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout)
            r.raise_for_status()
            return clean_html(r.text)
        except Exception:
            if attempt == retries:
                return ""
    return ""

def ensure_parent_dir(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)

def process_urls(input_path: str, output_path: str, workers: int, retries: int):
    urls = read_urls(input_path)
    print(f"读取到 {len(urls)} 个 URL")
    ensure_parent_dir(output_path)
    seen_hashes = set()
    with open(output_path, "w", encoding="utf-8") as fout, \
         ThreadPoolExecutor(max_workers=workers) as exe:
        futures = {exe.submit(download_one, u, retries): u for u in urls}
        for fut in tqdm(as_completed(futures), total=len(futures), desc="Downloading"):
            txt = fut.result()
            if not txt:
                continue
            h = hashlib.md5(txt.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            fout.write(txt + "\n")
    print(f"完成，已写入 {len(seen_hashes)} 条记录到 {output_path}")

def process_warc(input_path: str, output_path: str):
    print(f"读取 WARC 文件：{input_path}")
    ensure_parent_dir(output_path)
    seen_hashes = set()
    with open(output_path, "w", encoding="utf-8") as fout:
        for raw_html in tqdm(iter_warc_records(input_path), desc="WARC → 文本"):
            txt = clean_html(raw_html)
            if not txt:
                continue
            h = hashlib.md5(txt.encode("utf-8")).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)
            fout.write(txt + "\n")
    print(f"WARC 清洗完成，写入 {len(seen_hashes)} 条记录到 {output_path}")

def main():
    args = parse_args()
    inp = args.input
    outp = args.output
    if inp.endswith(('.warc', '.warc.gz', '.warc.gzip')):
        process_warc(inp, outp)
    else:
        process_urls(inp, outp, args.workers, args.retries)

if __name__ == "__main__":
    main()
