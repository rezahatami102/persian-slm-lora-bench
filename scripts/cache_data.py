import os

from farstail.datasets import farstail

# Download FarsTail once and store it locally so training never needs the network
train_df, val_df, test_df = farstail.load_original_data()

os.makedirs("data", exist_ok=True)
for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
    df.to_csv(f"data/farstail_{name}.csv", index=False, encoding="utf-8")
    print(name, len(df))