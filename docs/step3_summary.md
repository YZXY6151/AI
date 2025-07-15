# 第3大步骤：通用文本数据采集 汇总

本模块实现了从 URL 列表和 Common Crawl WARC 文件批量下载并清洗纯文本的完整流程，包括：

## 3.1 编写下载脚本
- 脚本路径：`scripts/data/download_texts.py`  
- 支持模式：
  - URL 列表：并发下载 + HTML 清洗 + MD5 去重  
  - WARC 文件：`warcio` 迭代 + HTML 清洗（lxml 解析）+ MD5 去重  
- 清洗流程：
  - 使用 `BeautifulSoup(..., "lxml")` 精确提取正文  
  - 忽略 `<script>` 和 `<style>` 标签内容  
  - 自动跳过无法解码或格式异常的数据块  
- 依赖：`requests`、`beautifulsoup4`、`lxml`、`tqdm`、`warcio`

## 3.2 测试提取结果
- 测试文件：`data/cc_warc/sample.warc.gz`（来自 CC-MAIN-2025-26 中的片段）  
- 运行脚本后，`data/cc.txt` 写入 7752 条纯文本记录  
- 验证：
  - `wc -l data/cc.txt` -> 7752  
  - `grep -E "<[A-Za-z/]" -n data/cc.txt` -> 无 HTML 标签  
  - `file -i data/cc.txt` -> 全部为 UTF-8 编码，无乱码

## 3.3 集成 cc_net
- 仓库：`external/cc_net`  
- 安装：`pip install -e external/cc_net`  
- 自定义配置文件：`test2.json`，设置了中文语言处理、4G 目标体积、最小长度 300 字符等  
- 示例配置内容（节选）：
  ```json
  {
    "config_name": "base",
    "dump": "CC-MAIN-2024-10",
    "output_dir": "data/cc_net_wiki",
    "lang_whitelist": ["zh"],
    "min_len": 300,
    "target_size": "4G"
  }
实际运行：

原始自动下载受限（403 Forbidden）

改用手动下载 .warc.gz，并通过 scripts/data/download_texts.py 处理

data/cc.txt 成功提取 7,752 条中文文本

3.4 文档与提交
汇总文档：docs/step3_summary.md

所有改动已提交，包含：

下载脚本改造为多模式支持

cc_net JSON 配置准备与测试

文本清洗效果验证与数据量记录

README 中 cc_net 用法说明已同步调整

