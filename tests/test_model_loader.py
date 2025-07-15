import pytest
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_load_local_yi_model():
    model_path = "models/yi-1.5-9b"
    # 加载 tokenizer
    tok   = AutoTokenizer.from_pretrained(model_path)
    # 加载模型
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=False)
    # 简单断言
    assert model.config is not None
    assert tok.vocab_size > 0
