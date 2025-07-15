#!/bin/bash
set -e

INPUT_PATH="wet.paths"
WET_DIR="wet_files"
OUTPUT_DIR="data"
MERGED_FILE="$OUTPUT_DIR/all_raw.txt"
PY_SCRIPT="wet_downloader.py"

mkdir -p "$WET_DIR"
mkdir -p "$OUTPUT_DIR"
> "$MERGED_FILE"  # 清空合并输出

echo "🔍 开始处理 wet.paths..."

while IFS= read -r line; do
    url="https://data.commoncrawl.org/${line}"
    fname=$(basename "$url")
    local_path="$WET_DIR/$fname"

    echo "⏳ 检查 $url"
    if wget --spider --quiet "$url"; then
        echo "✅ 可用：$url"
        if [ ! -f "$local_path" ]; then
            wget -q "$url" -P "$WET_DIR"
            echo "⬇️ 已下载 $fname"
        fi
        echo "🧽 清洗 $fname"
        # 生成唯一输出路径
        out_path="wet_files/outputs/${fname}.txt"
        mkdir -p wet_files/outputs

        python "$PY_SCRIPT" \
        --input "$local_path" \
        --output "$out_path"

        if [ -f "$out_path" ]; then
        cat "$out_path" >> "$MERGED_FILE"
        echo "✅ 合并成功：$out_path"
        else
        echo "❌ 警告：未生成输出 $out_path，跳过合并"
        fi

        rm -f tmp_output.txt
    else
        echo "❌ 跳过：$url 不存在"
    fi
done < "$INPUT_PATH"

echo "🎉 完成！所有文本已写入：$MERGED_FILE"
wc -l "$MERGED_FILE"
