import json

import pandas as pd

rows = [json.loads(line) for line in open("results/results.jsonl", encoding="utf-8")]
df = pd.DataFrame(rows)

group_cols = ["model_id", "lora_r", "lr"]
metrics = ["val_macro_f1", "test_accuracy", "test_macro_f1", "peak_vram_mib"]

summary = df.groupby(group_cols)[metrics].agg(["mean", "std", "count"])


def fmt(g):
    out = {"n_seeds": int(g[("test_macro_f1", "count")])}
    for m in metrics:
        mean, std = g[(m, "mean")], g[(m, "std")]
        if m == "peak_vram_mib":
            out[m] = f"{mean:.0f}"
        else:
            out[m] = f"{mean:.4f} ± {std:.4f}" if pd.notna(std) else f"{mean:.4f}"
    return pd.Series(out)


table = summary.apply(fmt, axis=1).reset_index()
print(table.to_markdown(index=False))

with open("results/summary.md", "w", encoding="utf-8") as f:
    f.write(table.to_markdown(index=False))