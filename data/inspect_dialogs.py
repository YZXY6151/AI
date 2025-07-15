import json

paths = [
    "data/dialogs_raw/personachat.jsonl",
    "data/dialogs_raw/dailydialog.jsonl",
    "data/dialogs_raw/go_emotions.jsonl",
]

def preview(path, n=3):
    print(f"\n--- Preview {path} ---")
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n: break
            obj = json.loads(line)
            # 美化打印，可根据数据集结构调整
            print(json.dumps(obj, ensure_ascii=False, indent=2))
    print("-" * 40)

if __name__ == "__main__":
    for p in paths:
        preview(p)
