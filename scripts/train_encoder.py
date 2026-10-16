import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

# FarsTail labels: e = entailment, n = neutral, c = contradiction
LABEL2ID = {"e": 0, "n": 1, "c": 2}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default="HooshvareLab/bert-fa-base-uncased")
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true", help="Tiny fast run for code checks")
    return p.parse_args()


def build_dataset(df, tokenizer, max_length):
    # BERT-style encoders take the sentence pair natively ([CLS] a [SEP] b [SEP])
    ds = Dataset.from_dict(
        {
            "premise": df["premise"].tolist(),
            "hypothesis": df["hypothesis"].tolist(),
            "labels": [LABEL2ID[x] for x in df["label"]],
        }
    )
    return ds.map(
        lambda b: tokenizer(
            b["premise"], b["hypothesis"], truncation=True, max_length=max_length
        ),
        batched=True,
        remove_columns=["premise", "hypothesis"],
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

    run_name = f"{args.model_id.split('/')[-1]}_full_lr{args.lr}_s{args.seed}"
    out_dir = os.path.join("outputs", run_name)

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
    train_ds = build_dataset(train_df, tokenizer, args.max_length)
    val_ds = build_dataset(val_df, tokenizer, args.max_length)
    test_ds = build_dataset(test_df, tokenizer, args.max_length)

    # Full fine-tuning (no LoRA, no quantization)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_id, num_labels=3
    )

    steps_per_epoch = -(-len(train_ds) // args.batch_size)
    warmup_steps = max(1, int(0.06 * steps_per_epoch * args.epochs))

    training_args = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=64,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        warmup_steps=warmup_steps,
        bf16=True,
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

    # lora_r = 0 marks full fine-tuning, so aggregate.py can group these runs
    record = {
        "run": run_name,
        "model_id": args.model_id,
        "lora_r": 0,
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