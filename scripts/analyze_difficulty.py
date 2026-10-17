import json

import pandas as pd
from sklearn.metrics import accuracy_score

# Same label ids as in training
LABEL2ID = {"e": 0, "n": 1, "c": 2}

test = pd.read_csv("data/farstail_test_difficulty.csv", encoding="utf-8")
y_true = test["label"].map(LABEL2ID).to_numpy()
is_easy = (test["difficulty"] == "easy").to_numpy()

# Run name -> configuration (first record wins, like aggregate.py).
# Runs without a logged record (e.g. smoke runs) are ignored on purpose.
configs = {}
with open("results/results.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        configs.setdefault(r["run"], (r["model_id"], r["lora_r"], r["lr"]))

rows = []
for run, (model_id, lora_r, lr) in configs.items():
    try:
        pred_df = pd.read_csv(f"results/predictions/{run}.csv")
    except FileNotFoundError:
        print("No predictions for", run)
        continue
    # Sanity check: saved labels must match the test file row by row
    assert (pred_df["label"].to_numpy() == y_true).all(), f"Label mismatch in {run}"
    pred = pred_df["pred"].to_numpy()
    easy_acc = accuracy_score(y_true[is_easy], pred[is_easy])
    hard_acc = accuracy_score(y_true[~is_easy], pred[~is_easy])
    rows.append(
        {
            "model_id": model_id,
            "lora_r": lora_r,
            "lr": lr,
            "easy_acc": easy_acc,
            "hard_acc": hard_acc,
            "gap": easy_acc - hard_acc,
        }
    )

df = pd.DataFrame(rows)


def mean_std(s):
    return f"{s.mean():.4f} ± {s.std():.4f}" if len(s) > 1 else f"{s.mean():.4f}"


out = []
for (model_id, lora_r, lr), g in df.groupby(["model_id", "lora_r", "lr"]):
    out.append(
        {
            "model_id": model_id,
            "lora_r": lora_r,
            "lr": lr,
            "n_seeds": len(g),
            "easy_acc": mean_std(g["easy_acc"]),
            "hard_acc": mean_std(g["hard_acc"]),
            "gap (easy - hard)": mean_std(g["gap"]),
        }
    )

summary = pd.DataFrame(out)
print(summary.to_markdown(index=False))
summary.to_markdown("results/difficulty_summary.md", index=False)