"""
verify_extraction.py — Confirm data/raw/images matches meta.jsonl

Checks:
  1. What the extracted folder structure actually looks like (flat vs nested)
  2. Whether img_id values from meta.jsonl resolve to real files
  3. Overall coverage: what % of meta.jsonl rows have a matching image on disk
  4. Whether the extracted file COUNT matches meta.jsonl row count (sanity check)
"""

import json
from pathlib import Path
from collections import defaultdict

EXTRACTED_ROOT = Path("data/raw/images")
META_PATH = Path("data/raw/meta.jsonl")


def main():
    # 1. Inspect top-level structure
    print("=" * 60)
    print("TOP-LEVEL STRUCTURE of", EXTRACTED_ROOT)
    print("=" * 60)
    top_level = list(EXTRACTED_ROOT.iterdir())
    print(f"Top-level entries: {len(top_level)}")
    for p in top_level[:10]:
        kind = "DIR" if p.is_dir() else "FILE"
        print(f"  [{kind}] {p.name}")
    if len(top_level) > 10:
        print(f"  ... and {len(top_level) - 10} more")

    # 2. Build a lookup: filename -> full path (one pass over the whole tree)
    print("\nBuilding filename -> path index (this walks the full tree once)...")
    filename_to_paths = defaultdict(list)
    total_files = 0
    for p in EXTRACTED_ROOT.rglob("*"):
        if p.is_file():
            total_files += 1
            filename_to_paths[p.name].append(p)
    print(f"Total files found on disk: {total_files:,}")

    # 3. Load meta.jsonl and check coverage
    print("\n" + "=" * 60)
    print("CHECKING img_id COVERAGE against meta.jsonl")
    print("=" * 60)

    total_rows = 0
    matched = 0
    unmatched_samples = []
    duplicate_name_samples = []

    with open(META_PATH, encoding="utf-8") as f:
        for line in f:
            total_rows += 1
            row = json.loads(line)
            img_id = row["img_id"]

            paths = filename_to_paths.get(img_id)
            if paths:
                matched += 1
                if len(paths) > 1 and len(duplicate_name_samples) < 5:
                    duplicate_name_samples.append((img_id, paths))
            else:
                if len(unmatched_samples) < 10:
                    unmatched_samples.append(img_id)

    print(f"Total meta.jsonl rows:     {total_rows:,}")
    print(f"Matched to a file on disk: {matched:,} ({matched/total_rows*100:.1f}%)")
    print(f"Unmatched:                 {total_rows - matched:,}")

    if unmatched_samples:
        print("\nSample unmatched img_ids (first 10):")
        for uid in unmatched_samples:
            print(f"  - {uid}")

    if duplicate_name_samples:
        print("\nWARNING: some img_id filenames appear in MULTIPLE locations "
              "(could mean nested-by-film structure with reused filenames, "
              "or actual duplicates):")
        for uid, paths in duplicate_name_samples:
            print(f"  {uid}:")
            for p in paths:
                print(f"    -> {p}")

    # 4. Show a few concrete example paths for the Dataset class
    print("\n" + "=" * 60)
    print("EXAMPLE RESOLVED PATHS (for building the Dataset class)")
    print("=" * 60)
    shown = 0
    with open(META_PATH, encoding="utf-8") as f:
        for line in f:
            if shown >= 5:
                break
            row = json.loads(line)
            img_id = row["img_id"]
            paths = filename_to_paths.get(img_id)
            if paths:
                print(f"  img_id={img_id}  ->  {paths[0]}")
                shown += 1

    print("\nDone. Use the coverage % above to decide if extraction is "
          "complete/correct before building the Dataset class.")


if __name__ == "__main__":
    main()