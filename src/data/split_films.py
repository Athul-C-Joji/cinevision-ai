import json
import random
import pandas as pd

meta_path = r"C:\Users\ATHUL C JOJI\.cache\huggingface\hub\datasets--Vchitect--ShotQA\snapshots\5043db350ec3804ae6208aa18bde93eb35841509\meta.jsonl"

records = []
with open(meta_path, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

# Get unique films and shuffle them (with a fixed seed for reproducibility)
films = df["title"].unique().tolist()
random.seed(42)
random.shuffle(films)

n = len(films)
train_end = int(n * 0.70)
val_end = int(n * 0.85)

train_films = set(films[:train_end])
val_films = set(films[train_end:val_end])
test_films = set(films[val_end:])

print(f"Total films: {n}")
print(f"Train films: {len(train_films)}")
print(f"Val films:   {len(val_films)}")
print(f"Test films:  {len(test_films)}")

df["split"] = df["title"].apply(
    lambda t: "train" if t in train_films else ("val" if t in val_films else "test")
)

print("\nRows per split:")
print(df["split"].value_counts())

# Save the split assignment so it's reproducible and reusable everywhere
df[["title", "img_id", "split"]].to_csv("data/processed/film_splits.csv", index=False)
print("\nSaved to data/processed/film_splits.csv")