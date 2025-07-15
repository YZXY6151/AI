import os
import requests
import gzip
from tqdm import tqdm

# 路径设置
BASE_URL = "https://data.commoncrawl.org/"
WET_LIST_PATH = "wet.paths"
WET_DIR = "wet_files"
OUTPUT_TXT_PATH = "data/cleaned_output.txt"

# 创建所需目录
os.makedirs(WET_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_TXT_PATH), exist_ok=True)

# 读取 wetpaths
with open(WET_LIST_PATH, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

urls = [BASE_URL + path for path in lines]

# 打开 output 文件
with open(OUTPUT_TXT_PATH, "a", encoding="utf-8") as f_out:
    for url in tqdm(urls, desc="批量下载 + 解压", unit="file"):
        filename = url.split("/")[-1]
        local_gz_path = os.path.join(WET_DIR, filename)

        # 如果 .gz 已存在，跳过下载
        if not os.path.exists(local_gz_path):
            try:
                r = requests.get(url, stream=True, timeout=30)
                if r.status_code != 200:
                    print(f"❌ 跳过无效地址: {url}")
                    continue

                with open(local_gz_path, "wb") as f:
                    for chunk in r.iter_content(1024 * 1024):
                        f.write(chunk)

            except Exception as e:
                print(f"❌ 下载失败: {url}，错误: {e}")
                continue

        # 解压并提取文本内容到 cleaned_output.txt
        try:
            with gzip.open(local_gz_path, "rt", encoding="utf-8", errors="ignore") as f_in:
                for line in f_in:
                    f_out.write(line)
        except Exception as e:
            print(f"⚠️ 解压失败: {filename}，错误: {e}")
            continue
