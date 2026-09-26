"""
Extract the camera-movement labeled dataset from grpo.json (the "videos" key
entries), validate the video files exist on disk, and report the class
distribution of movement labels.

Run from the project root with the venv active:
    python -m src.data.extract_movement_labels
"""

import csv
import json
import os
import re
from collections import Counter

DATA_DIR = os.path.join("data", "raw")
GRPO_PATH = os.path.join(DATA_DIR, "grpo.json")
IMAGES_DIR = os.path.join(DATA_DIR, "images")  # videos live here too (flat structure)

ANSWER_RE = re.compile(r"<answer>\s*([A-D])\s*</answer>", re.IGNORECASE)
OPTIONS_RE = re.compile(r"([A-D])\.\s*([^\n]+)")


def main():
    with open(GRPO_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    movement_entries = [e for e in data if isinstance(e, dict) and "videos" in e]
    print(f"Total grpo.json entries: {len(data)}")
    print(f"Entries with a 'videos' key: {len(movement_entries)}")

    missing_files = []
    label_counts = Counter()
    parse_failures = 0
    rows = []

    for e in movement_entries:
        vids = e.get("videos", [])
        if not vids:
            continue
        rel_path = vids[0]
        fname = os.path.basename(rel_path)
        full_path = os.path.join(IMAGES_DIR, fname)

        if not os.path.exists(full_path):
            missing_files.append(fname)

        question_text = ""
        for m in e.get("messages", []):
            if m.get("role") == "user":
                question_text = m.get("content", "")
                break

        solution = e.get("solution", "")
        ans_match = ANSWER_RE.search(solution)
        letter = ans_match.group(1).upper() if ans_match else None

        options = dict(OPTIONS_RE.findall(question_text))
        label = options.get(letter, "UNKNOWN") if letter else "UNKNOWN"

        if letter is None or label == "UNKNOWN":
            parse_failures += 1

        label_counts[label] += 1
        rows.append({"file": fname, "label": label})

    print(f"\nVideo files referenced but missing on disk: {len(missing_files)}")
    if missing_files:
        print("  First few missing:", missing_files[:5])

    print(f"\nEntries where label parsing failed: {parse_failures}")

    print(f"\nMovement label class distribution ({len(label_counts)} classes):")
    for label, count in label_counts.most_common():
        print(f"  {label:20s} {count:5d}")

    out_csv = os.path.join("data", "processed", "movement_labels.csv")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "label"])
        for r in rows:
            writer.writerow([r["file"], r["label"]])
    print(f"\nSaved {len(rows)} rows to {out_csv}")


if __name__ == "__main__":
    main()