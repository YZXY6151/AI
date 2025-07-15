import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def load_model(model_path: str):
    """
    加载本地模型并返回 (model, tokenizer)。
    - model_path: 本地模型目录，如 "models/yi-1.5-9b"
    """
    # 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True
    )
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        local_files_only=True
    )
    # 开启半精度与 cuDNN benchmark
    model.half()
    torch.backends.cudnn.benchmark = True
    model.eval()
    return model, tokenizer
