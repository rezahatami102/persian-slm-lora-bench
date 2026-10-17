import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

train, val, test = (
    pd.read_csv(f"data/farstail_{s}.csv", encoding="utf-8")
    for s in ("train", "val", "test")
)

# Hypothesis-only classifier: it never sees the premise, so any success
# comes from superficial cues in the hypothesis text.
vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
clf = LogisticRegression(max_iter=2000, C=3.0)
clf.fit(vec.fit_transform(train["hypothesis"]), train["label"])

for name, df in [("val", val), ("test", test)]:
    pred = clf.predict(vec.transform(df["hypothesis"]))
    print(f"{name} hypothesis-only accuracy: {accuracy_score(df['label'], pred):.4f}")

# Easy = the hypothesis-only model gets it right; hard = it gets it wrong
test = test.copy()
test_pred = clf.predict(vec.transform(test["hypothesis"]))
test["difficulty"] = ["easy" if p == y else "hard" for p, y in zip(test_pred, test["label"])]

print(test["difficulty"].value_counts())
print(pd.crosstab(test["difficulty"], test["label"]))

# Row order is unchanged, so it matches the test set used in training
test.to_csv("data/farstail_test_difficulty.csv", index=False, encoding="utf-8")