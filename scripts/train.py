import argparse
import json
import os

import numpy as np
import torch
from datasets import Dataset
import pandas as pd
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# FarsTail labels: e = entailment, n = neutral, c = contradiction
LABEL2ID = {"e": 0, "n": 1, "c": 2}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="Qwen/Qwen2.5-0.5B")
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--max_length", type=int, default=448)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="Tiny fast run for code checks")
    return p.parse_args()


def build_dataset(df, tokenizer, max_length):
    # Explicit template so the model can tell premise and hypothesis apart.
    # The last token (":") is what the classification head reads.
    texts = [
        f"Premise: {p}\nHypothesis: {h}\nRelation:"
        for p, h in zip(df["premise"], df["hypothesis"])
    ]
    labels = [LABEL2ID[x] for x in df["label"]]
    ds = Dataset.from_dict({"text": texts, "labels": labels})
    return ds.map(
        lambda batch: tokenizer(batch["text"], truncation=True, max_length=max_length),
        batched=True,
        remove_columns=["text"],
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    run_name = f"{args.model_id.split('/')[-1]}_r{args.lora_r}_lr{args.lr}_s{args.seed}"
    out_dir = os.path.join("outputs", run_name)

    # Load from the local cache created by scripts/cache_data.py (no network needed)
    train_df, val_df, test_df = (
        pd.read_csv(f"data/farstail_{s}.csv", encoding="utf-8")
        for s in ("train", "val", "test")
    )

    if args.smoke:
        train_df = train_df.sample(256, random_state=0)
        val_df = val_df.head(64)
        test_df = test_df.head(64)
        args.epochs = 1

    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = build_dataset(train_df, tokenizer, args.max_length)
    val_ds = build_dataset(val_df, tokenizer, args.max_length)
    test_ds = build_dataset(test_df, tokenizer, args.max_length)

    # 4-bit NF4 quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id,
        num_labels=3,
        quantization_config=bnb_config,
        device_map={"": 0},
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=2 * args.lora_r,
        lora_dropout=0.05,
        task_type="SEQ_CLS",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Warmup = 3% of total optimizer steps (computed manually for version compatibility)
    steps_per_epoch = -(-len(train_ds) // (args.batch_size * args.grad_accum))
    warmup_steps = max(1, int(0.03 * steps_per_epoch * args.epochs))

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=32,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        logging_steps=50,
        report_to="none",
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    val_metrics = trainer.evaluate(val_ds, metric_key_prefix="val")
    test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")

    record = {
        "run": run_name,
        "model_id": args.model_id,
        "lora_r": args.lora_r,
        "lr": args.lr,
        "epochs": args.epochs,
        "seed": args.seed,
        "val_accuracy": val_metrics["val_accuracy"],
        "val_macro_f1": val_metrics["val_macro_f1"],
        "test_accuracy": test_metrics["test_accuracy"],
        "test_macro_f1": test_metrics["test_macro_f1"],
        "peak_vram_mib": torch.cuda.max_memory_allocated() / 1024**2,
    }
    print(json.dumps(record, indent=2))

    if args.smoke:
        print("Smoke run finished; results not saved.")
        return

    os.makedirs("results", exist_ok=True)
    with open("results/results.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()