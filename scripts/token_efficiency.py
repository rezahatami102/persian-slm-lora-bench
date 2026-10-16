import pandas as pd
from transformers import AutoTokenizer

df = pd.read_csv("data/farstail_train.csv", encoding="utf-8")

# Same prompt template used in training
texts = [
    f"Premise: {p}\nHypothesis: {h}\nRelation:"
    for p, h in zip(df["premise"], df["hypothesis"])
]
total_chars = sum(len(t) for t in texts)

for model_id in ["Qwen/Qwen2.5-0.5B", "google/gemma-3-1b-pt"]:
    tok = AutoTokenizer.from_pretrained(model_id)
    lengths = [len(tok(t)["input_ids"]) for t in texts]
    print(
        f"{model_id}: mean tokens = {sum(lengths) / len(lengths):.1f}, "
        f"max = {max(lengths)}, chars/token = {total_chars / sum(lengths):.2f}"
    )