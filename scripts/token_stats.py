import numpy as np
from transformers import AutoTokenizer
from farstail.datasets import farstail

train, _, _ = farstail.load_original_data()

model_id = "Qwen/Qwen2.5-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Tokenize each (premise, hypothesis) pair the way the model will see it
lengths = [
    len(tokenizer(p, h)["input_ids"])
    for p, h in zip(train["premise"], train["hypothesis"])
]

lengths = np.array(lengths)
print("Mean:", lengths.mean())
for q in (50, 90, 95, 99, 100):
    print(f"P{q}:", int(np.percentile(lengths, q)))