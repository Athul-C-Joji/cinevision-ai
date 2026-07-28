import json
import pandas as pd

meta_path = r"C:\Users\ATHUL C JOJI\.cache\huggingface\hub\datasets--Vchitect--ShotQA\snapshots\5043db350ec3804ae6208aa18bde93eb35841509\meta.jsonl"

records = []
with open(meta_path, "r", encoding="utf-8") as f:
    for line in f:
        records.append(json.loads(line))

df = pd.DataFrame(records)

dimensions = ["Frame Size", "Shot Type", "Lens Size", "Lighting", "Lighting Type", "Composition"]

for col in dimensions:
    non_null = df[col].dropna()
    n_multi = non_null.apply(lambda x: "," in x).sum()
    pct_multi = round(100 * n_multi / len(non_null), 1)
    all_tags = set()
    for val in non_null:
        for tag in val.split(","):
            all_tags.add(tag.strip())
    print(f"{col}: {pct_multi}% multi-tagged | {len(all_tags)} unique individual tags")
    print(f"  Sample tags: {sorted(all_tags)[:15]}")
    print()