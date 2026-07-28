import json
import pandas as pd

meta_path = r"C:\Users\ATHUL C JOJI\.cache\huggingface\hub\datasets--Vchitect--ShotQA\snapshots\5043db350ec3804ae6208aa18bde93eb35841509\meta.jsonl"

records = []
with open(meta_path, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)
print("Total rows:", len(df))
print("\nColumns:", list(df.columns))
print("\n% non-null per column:")
print((df.notna().mean() * 100).round(1).sort_values(ascending=False))

print("\n--- Key dimension value counts ---")
for col in ["Frame Size", "Shot Type", "Lens Size", "Lighting", "Lighting Type", "Composition"]:
    print(f"\n{col}:")
    print(df[col].value_counts().head(10))

print("\nUnique films (by title):", df["title"].nunique())