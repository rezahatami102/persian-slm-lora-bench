import json

import pandas as pd

with open("results/results.jsonl", encoding="utf-8") as f:
    df = pd.DataFrame([json.loads(line) for line in f])

# Older records have no train_frac column: treat them as full-data runs
if "train_frac" not in df:
    df["train_frac"] = 1.0
df["train_frac"] = df["train_frac"].fillna(1.0)

# Keep only the first record for each unique configuration
df = df.drop_duplicates(
    subset=["model_id", "lora_r", "lr", "epochs", "train_frac", "seed"], keep="first"
)

group_cols = ["model_id", "lora_r", "lr", "train_frac"]
metrics = ["val_macro_f1", "test_accuracy", "test_macro_f1", "peak_vram_mib"]

rows = []
for keys, g in df.groupby(group_cols):
    row = dict(zip(group_cols, keys))
    row["n_seeds"] = len(g)
    for m in metrics:
        if m == "peak_vram_mib":
            row[m] = f"{g[m].mean():.0f}"
        elif len(g) > 1:
            row[m] = f"{g[m].mean():.4f} ± {g[m].std():.4f}"
        else:
            row[m] = f"{g[m].mean():.4f}"
    rows.append(row)

# Sort on the numeric columns (before they were turned into text)
table = pd.DataFrame(rows).sort_values(["model_id", "train_frac", "lora_r", "lr"])
print(table.to_markdown(index=False))

with open("results/summary.md", "w", encoding="utf-8") as f:
    f.write(table.to_markdown(index=False))