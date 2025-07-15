#!/usr/bin/env python3
import os
import json
import torch
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset
from transformers import EarlyStoppingCallback, set_seed

# ─── 环境清理：强制单卡运行 ───
os.environ["CUDA_VISIBLE_DEVICES"]        = "0"
os.environ["LOCAL_RANK"]                 = "0"
os.environ["RANK"]                       = "0"
os.environ["WORLD_SIZE"]                 = "1"
os.environ["ACCELERATE_DISABLE_MAPPING"] = "true"
os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
os.environ.setdefault("MASTER_PORT", "29500")

def main():
    # ─── 路径配置 ───
    model_name       = "models/yi-1.5-9b"
    lora_config_path = "configs/lora_config.json"
    train_file       = "data/finetune/train.jsonl"
    valid_file       = "data/finetune/valid.jsonl"
    output_dir       = "models/yi-lora"

    # —— 固定随机种子，保证可复现 —— 
    set_seed(42)

    # ─── Tokenizer & Base Model ───
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    # 把 pad_token 设置成 eos，以便后续 collator 能正确 pad
    tokenizer.pad_token = tokenizer.eos_token

    sep = tokenizer.eos_token or "\n"

    bnb_config = BitsAndBytesConfig(load_in_8bit=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=None
    )
    base_model = prepare_model_for_kbit_training(base_model)

    print(f"🚀 当前显存使用: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # ─── 加载 & 注入 LoRA ───
    with open(lora_config_path) as f:
        lora_cfg = LoraConfig(**json.load(f))
    model = get_peft_model(base_model, lora_cfg)

    # ─── Trainable 参数占比 ───
    total, trainable = 0, 0
    for _, p in model.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"🚀 Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")

    # ─── 加载数据集 & Tokenize（只输出 input_ids, attention_mask） ───
    ds = load_dataset("json", data_files={"train": train_file, "validation": valid_file})

    def tokenize_fn(example):
        text = example["prompt"] + sep + example["completion"]
        return tokenizer(text, truncation=True, max_length=512)

    train_ds = ds["train"].map(
        tokenize_fn,
        batched=False,
        remove_columns=["prompt", "completion", "metadata"]
    )
    eval_ds = ds["validation"].map(
        tokenize_fn,
        batched=False,
        remove_columns=["prompt", "completion", "metadata"]
    )

    # ─── 使用 DataCollatorForLanguageModeling (mlm=False) ───
    # 自动复制 inputs 到 labels 并做动态 pad，保证三者对齐 :contentReference[oaicite:0]{index=0}
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # ─── 训练参数 ───
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        learning_rate=5e-5,               # 低 LR，平滑收敛
        warmup_steps=300,                 # 更长 warmup
        lr_scheduler_type="linear",       # 线性衰减
        num_train_epochs=5,               # 多跑几轮，用 EarlyStopping 决定何时停
        logging_steps=1,
        weight_decay=0.01,            # L2 正则，防止过拟合
        max_grad_norm=1.0,            # 梯度裁剪，避免梯度爆炸
        dataloader_drop_last=True,        # 保证 batch 大小一致
        load_best_model_at_end=True,      # 训练结束后自动加载 “最佳” checkpoint
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy="steps",
        save_steps=500,                   # 稍微减少 checkpoint 频率
        save_total_limit=3,  
        eval_strategy="steps",
        eval_steps=500,
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
        report_to="none",
        label_names=["labels"],  # Trainer 用来计算 loss :contentReference[oaicite:1]{index=1}
    )

    # ─── Trainer ───
    trainer = Trainer(
        model=model,
        eval_dataset=eval_ds, 
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer,         # 让 Trainer 自动应用 tokenizer 的 dynamic padding :contentReference[oaicite:2]{index=2}
        data_collator=data_collator,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
    )

    # ─── 启动训练 ───
    trainer.train()
    trainer.save_model(output_dir)
    print("✅ LoRA 微调完成，模型已保存在", output_dir)


if __name__ == "__main__":
    main()
