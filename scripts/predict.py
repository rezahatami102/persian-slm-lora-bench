import gc
import glob
import json
import os

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)

# Reuse the exact data builders used in training (prompt template, max length)
from train import build_dataset as build_causal_dataset
from train_encoder import build_dataset as build_encoder_dataset


def best_checkpoint(run_dir):
    ckpts = sorted(
        glob.glob(os.path.join(run_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[1]),
    )
    if not ckpts:
        return None
    with open(os.path.join(ckpts[-1], "trainer_state.json"), encoding="utf-8") as f:
        state = json.load(f)
    best = state.get("best_model_checkpoint")
    if best:
        # Stored path may be absolute: keep only the folder name
        path = os.path.join(run_dir, os.path.basename(best.replace("\\", "/")))
        if os.path.isdir(path):
            return path
    return ckpts[-1]


def load_model(ckpt):
    adapter_cfg = os.path.join(ckpt, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        # LoRA run: 4-bit base model + saved adapter (and classification head)
        with open(adapter_cfg, encoding="utf-8") as f:
            base_id = json.load(f)["base_model_name_or_path"]
        tokenizer = AutoTokenizer.from_pretrained(base_id)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            base_id, num_labels=3, quantization_config=bnb_config, device_map={"": 0}
        )
        model.config.pad_token_id = tokenizer.pad_token_id
        model = PeftModel.from_pretrained(model, ckpt)
        return model, tokenizer, "causal"
    # Full fine-tuned encoder run
    tokenizer = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt).to("cuda")
    return model, tokenizer, "encoder"


@torch.no_grad()
def predict(model, tokenizer, ds):
    model.eval()
    loader = DataLoader(
        ds.remove_columns("labels"),
        batch_size=32,
        collate_fn=DataCollatorWithPadding(tokenizer),
    )
    preds = []
    for batch in loader:
        batch = {k: v.to("cuda") for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(**batch).logits
        preds.append(logits.argmax(-1).cpu())
    return torch.cat(preds).numpy()


def main():
    test_df = pd.read_csv("data/farstail_test.csv", encoding="utf-8")

    # Logged test Macro-F1 per run name, used to verify the recomputed value
    logged = {}
    with open("results/results.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            logged.setdefault(r["run"], []).append(r["test_macro_f1"])

    os.makedirs("results/predictions", exist_ok=True)
    for run_dir in sorted(glob.glob("outputs/*")):
        run = os.path.basename(run_dir)
        out_path = f"results/predictions/{run}.csv"
        if os.path.exists(out_path):
            continue
        ckpt = best_checkpoint(run_dir)
        if ckpt is None:
            continue

        model, tokenizer, kind = load_model(ckpt)
        if kind == "causal":
            ds = build_causal_dataset(test_df, tokenizer, 448)
        else:
            ds = build_encoder_dataset(test_df, tokenizer, 256)

        preds = predict(model, tokenizer, ds)
        y_true = np.array(ds["labels"])
        f1 = f1_score(y_true, preds, average="macro")
        ref = logged.get(run, [])
        status = "OK" if any(abs(f1 - x) < 0.005 for x in ref) else "MISMATCH"
        print(f"{run}: recomputed = {f1:.4f} | logged = {[round(x, 4) for x in ref]} | {status}")

        pd.DataFrame(
            {"row": range(len(preds)), "label": y_true, "pred": preds}
        ).to_csv(out_path, index=False)

        del model
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()