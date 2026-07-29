"""
validate_images.py — Find all broken/corrupt image files before training

Scans every img_id referenced in film_splits.csv, tries to actually open
each file with PIL (not just check it exists), and reports any that fail.

This catches corrupted downloads, truncated files, or non-image content
saved with an image extension -- all of which crash training if not
caught ahead of time.

Usage:
    python validate_images.py

Outputs:
    reports/broken_images.csv  -- list of broken img_ids + error message
"""

from pathlib import Path
import pandas as pd
from PIL import Image

SPLITS_CSV = Path("data/processed/film_splits.csv")
IMAGES_DIR = Path("data/raw/images")
OUTPUT_CSV = Path("reports/broken_images.csv")


def main():
    OUTPUT_CSV.parent.mkdir(exist_ok=True)

    df = pd.read_csv(SPLITS_CSV)
    img_ids = df["img_id"].unique()
    print(f"Checking {len(img_ids):,} unique images...")

    broken = []
    checked = 0

    for img_id in img_ids:
        checked += 1
        path = IMAGES_DIR / img_id

        if not path.exists():
            broken.append((img_id, "FILE_NOT_FOUND"))
            continue

        try:
            with Image.open(path) as img:
                img.verify()  # checks file integrity without fully decoding
            # verify() closes the file handle, need to reopen to actually
            # load pixel data -- verify() alone misses some truncation
            # issues that only surface on full decode.
            with Image.open(path) as img:
                img.convert("RGB").load()
        except Exception as e:
            broken.append((img_id, f"{type(e).__name__}: {e}"))

        if checked % 5000 == 0:
            print(f"  ...{checked:,}/{len(img_ids):,} checked, "
                  f"{len(broken)} broken so far")

    print(f"\nDone. {len(broken)} broken out of {len(img_ids):,} total "
          f"({len(broken)/len(img_ids)*100:.2f}%)")

    if broken:
        broken_df = pd.DataFrame(broken, columns=["img_id", "error"])
        broken_df.to_csv(OUTPUT_CSV, index=False)
        print(f"Broken file list saved to: {OUTPUT_CSV}")
        print("\nFirst few broken files:")
        for img_id, err in broken[:10]:
            print(f"  {img_id}: {err}")
    else:
        print("No broken images found!")


if __name__ == "__main__":
    main()