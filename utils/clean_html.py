from bs4 import BeautifulSoup
import re

def clean_html(raw: str) -> str:
    """
    去除 HTML 中的脚本、样式、noscript 等标签，提取纯文本，
    并合并多余空白为单个空格。
    """
    soup = BeautifulSoup(raw, "html.parser")
    # 移除脚本和样式等
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    # 提取文本
    text = soup.get_text(separator="\n")
    # 合并空白
    return re.sub(r"\s+", " ", text).strip()

# 本地简单测试
if __name__ == "__main__":
    sample = "<html><body><script>alert(1)</script><p>Hello   world</p></body></html>"
    print(clean_html(sample))  # 期望输出: "Hello world"
