import json

meta_path = r"C:\Users\ATHUL C JOJI\.cache\huggingface\hub\datasets--Vchitect--ShotQA\snapshots\5043db350ec3804ae6208aa18bde93eb35841509\meta.jsonl"

with open(meta_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        entry = json.loads(line)
        print(json.dumps(entry, indent=2))
        print("---")
        if i >= 2:
            break