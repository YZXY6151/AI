# 通用文本下载脚本说明

本目录包含一个用于从网页批量下载并清洗纯文本的脚本，以及下载结果。

## 脚本路径
- `scripts/data/download_texts.py`

## 参数说明
- `--input` 或 `-i`：URL 列表文件路径（每行一个 URL）  
- `--output` 或 `-o`：输出纯文本文件路径，默认 `data/raw.txt`  
- `--workers` 或 `-w`：并发下载线程数，默认 4  
- `--retries` 或 `-r`：每个 URL 最大重试次数，默认 3  

## 使用示例
```bash
# 使用 4 个线程、每个 URL 最多重试 3 次
python scripts/data/download_texts.py \
  --input test_urls.txt \
  --output data/raw.txt \
  --workers 4 \
  --retries 3

---

# ## cc_net 集成

# 使用 cc_net 从 Wikipedia 或 WARC 提取额外文本：

# ```bash
# # 安装（在 external/cc_net 目录下）
# cd external/cc_net
# pip install -e .

# # 从 Wikipedia dump 提取 English 文本
# python -m cc_net.mine -d wikipedia --lang_whitelist '["en"]' -o data/cc.txt

# # 或者处理 Common Crawl WARC 文件
# python scripts/data/download_texts.py \
#   --input data/cc_warc/sample.warc.gz \
#   --output data/cc.txt

# 安装 cc_net（在 external/cc_net 目录下）
cd external/cc_net
pip install -e .

# 提取英文 Wikipedia 纯文本
python -m cc_net.mine -d wikipedia --lang_whitelist '["en"]' -o data/cc_net_wiki

# 安装依赖（在 ai-env 环境中）
pip install warcio beautifulsoup4 lxml tqdm

# 下载 WARC 文件（示例）
mkdir -p data/cc_warc
wget https://data.commoncrawl.org/crawl-data/CC-MAIN-2025-26/segments/1749709481111.44/warc/CC-MAIN-20250612112840-20250612142840-00000.warc.gz \
     -O data/cc_warc/sample.warc.gz

# 清洗 WARC 内容，提取纯文本
python scripts/data/download_texts.py \
  --input data/cc_warc/sample.warc.gz \
  --output data/cc.txt




第五步
## 对话与情感数据集

- **personachat_truecased**  
  路径：`data/dialogs_raw/personachat.jsonl`  
  描述：PersonaChat（真值大小写版），含多条 persona 描述和候选回复。

- **daily_dialog**  
  路径：`data/dialogs_raw/dailydialog.jsonl`  
  描述：DailyDialog 对话数据，含 act 与 emotion 标签。

- **go_emotions**  
  路径：`data/dialogs_raw/go_emotions.jsonl`  
  描述：GoEmotions 情感分类数据，含 27 种情绪标签。
