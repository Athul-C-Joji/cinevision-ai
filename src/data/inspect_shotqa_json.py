"""
Quick inspection of sft.json, sft_v1.1.json, grpo.json from ShotQA.
Goal: understand schema, and find whether/how the .webm_*.mp4 video clips
(found in data/raw/images/) are referenced and labeled.

Run from the project root (D:\\CineVision AI\\cinevision-ai) with the venv active:
    python inspect_json_files.py
"""

import json
import os

DATA_DIR = os.path.join("data", "raw")

FILES = ["sft.json", "grpo.json", "sft_v1.1.json"]  # smallest first


def inspect(path):
    print(f"\n{'=' * 70}")
    print(f"Inspecting: {path}")
    print("=" * 70)

    if not os.path.exists(path):
        print("  FILE NOT FOUND — skipping")
        return

    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  Size: {size_mb:.1f} MB")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"  Top-level type: {type(data).__name__}")

    if isinstance(data, list):
        print(f"  Number of entries: {len(data)}")
        if len(data) > 0:
            sample = data[0]
            print(f"  First entry type: {type(sample).__name__}")
            if isinstance(sample, dict):
                print(f"  First entry keys: {list(sample.keys())}")
                print(f"  First entry (pretty):")
                print(json.dumps(sample, indent=2, ensure_ascii=False)[:2000])
    elif isinstance(data, dict):
        print(f"  Top-level keys: {list(data.keys())}")

    # Search for any reference to .webm_ (the video clip naming pattern)
    # Fast full scan: only check the 'images' field of every entry (not the
    # whole message tree), so this covers ALL entries even in the 780k-row file.
    print("\n  Searching ALL entries' 'images' field for '.webm_' references...")
    webm_hits = []
    non_jpg_hits = []

    if isinstance(data, list):
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                continue
            imgs = entry.get("images", [])
            if not isinstance(imgs, list):
                imgs = [imgs]
            for img in imgs:
                if not isinstance(img, str):
                    continue
                if ".webm_" in img:
                    if len(webm_hits) < 5:
                        webm_hits.append((i, img))
                elif not img.lower().endswith(".jpg"):
                    if len(non_jpg_hits) < 5:
                        non_jpg_hits.append((i, img))

    print(f"  Total .webm_ hits found: {'5+ (capped sample below)' if len(webm_hits) == 5 else len(webm_hits)}")
    for i, v in webm_hits:
        print(f"    entry[{i}].images = {v}")

    if non_jpg_hits:
        print(f"\n  Also found non-.jpg image refs (other formats):")
        for i, v in non_jpg_hits:
            print(f"    entry[{i}].images = {v}")

    # Check for multi-image entries (possible frame-sequence encoding for
    # temporal/movement questions) and any mention of "movement" in question text.
    print("\n  Checking for multi-image entries (frame sequences)...")
    multi_image_count = 0
    multi_image_sample = None
    movement_hits = 0
    movement_sample = None

    if isinstance(data, list):
        for i, entry in enumerate(data):
            if not isinstance(entry, dict):
                continue
            imgs = entry.get("images", [])
            if isinstance(imgs, list) and len(imgs) > 1:
                multi_image_count += 1
                if multi_image_sample is None:
                    multi_image_sample = (i, entry)

            msgs = entry.get("messages", [])
            for m in msgs:
                content = m.get("content", "") if isinstance(m, dict) else ""
                if isinstance(content, str) and "movement" in content.lower():
                    movement_hits += 1
                    if movement_sample is None:
                        movement_sample = (i, entry)
                    break

    print(f"  Entries with >1 image: {multi_image_count}")
    if multi_image_sample:
        i, e = multi_image_sample
        print(f"  Sample multi-image entry [{i}]:")
        print(json.dumps(e, indent=2, ensure_ascii=False)[:1500])

    print(f"\n  Entries mentioning 'movement' in question text: {movement_hits}")
    if movement_sample:
        i, e = movement_sample
        print(f"  Sample movement entry [{i}]:")
        print(json.dumps(e, indent=2, ensure_ascii=False)[:1500])


if __name__ == "__main__":
    for fname in FILES:
        inspect(os.path.join(DATA_DIR, fname))