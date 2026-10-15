from farstail.datasets import farstail

train, val, test = farstail.load_original_data()

# Column names and dtypes
print(train.dtypes)

# Label distribution in each split
for name, df in [("train", train), ("val", val), ("test", test)]:
    print(f"\n{name} label counts:")
    print(df["label"].value_counts())

# One full row, printed field by field
row = train.iloc[0]
for col in train.columns:
    print(f"\n[{col}]\n{row[col]}")

# Text length statistics (characters)
print("\nPremise length:", train["premise"].str.len().describe())
print("\nHypothesis length:", train["hypothesis"].str.len().describe())