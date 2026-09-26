"""
src/data/split_movement_clips.py

Film-level style split for the camera-movement dataset: groups movement
clips by their source video (not by clip filename) so that clips cut from
the same source video never leak across train/val/test -- same principle
as split_films.py, applied here since many clips share a source video
(e.g. "fSr_nfsdlRg.webm_66.mp4" and "fSr_nfsdlRg.webm_73.mp4" are two
different clips from the same source).

Reads data/processed/movement_labels_encoded.csv, excludes clips with no
extracted frames (per reports/movement_frame_extraction.csv), groups by
base source-video ID parsed from the filename, and does a 70/15/15 split
by that group with a fixed random seed.

Usage:
    python -m src.data.split_movement_clips
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS_CSV = PROJECT_ROOT / "data" / "processed" / "movement_labels_encoded.csv"
EXTRACTION_REPORT = PROJECT_ROOT / "reports" / "movement_frame_extraction.csv"
OUT_CSV = PROJECT_ROOT / "data" / "processed" / "movement_splits.csv"

# Matches e.g. "uaKXU_zuRGM.webm_24.mp4" -> base id "uaKXU_zuRGM"
# or "SbozyYNrWmc.mkv_30.mp4" -> base id "SbozyYNrWmc"
BASE_ID_RE = re.compile(r"^(.+)\.\w+_\d+\.mp4$")


def get_base_video_id(filename: str) -> str:
    m = BASE_ID_RE.match(filename)
    if m:
        return m.group(1)
    return filename  # fallback: treat as its own group


def main():
    df = pd.read_csv(LABELS_CSV)

    if EXTRACTION_REPORT.exists():
        ext_df = pd.read_csv(EXTRACTION_REPORT)
        failed = set(ext_df[ext_df["status"] == "DECODE_FAILED"]["filename"])
        before = len(df)
        df = df[~df["filename"].isin(failed)].reset_index(drop=True)
        print(f"Excluded {before - len(df)} clips with no extracted frames")
    else:
        print(f"WARNING: {EXTRACTION_REPORT} not found -- "
              f"not filtering failed-extraction clips")

    df["base_video_id"] = df["filename"].apply(get_base_video_id)

    unmatched = df[df["base_video_id"] == df["filename"]]
    if len(unmatched) > 0:
        print(f"WARNING: {len(unmatched)} filenames didn't match the expected "
              f"pattern and are being treated as their own group:")
        print(unmatched["filename"].tolist())

    groups = df["base_video_id"].unique().tolist()
    rng = np.random.default_rng(SEED)
    rng.shuffle(groups)

    n = len(groups)
    n_train = int(round(n * SPLIT_RATIOS["train"]))
    n_val = int(round(n * SPLIT_RATIOS["val"]))

    train_groups = set(groups[:n_train])
    val_groups = set(groups[n_train:n_train + n_val])
    test_groups = set(groups[n_train + n_val:])

    def assign_split(base_id):
        if base_id in train_groups:
            return "train"
        elif base_id in val_groups:
            return "val"
        else:
            return "test"

    df["split"] = df["base_video_id"].apply(assign_split)

    out_df = df[["filename", "base_video_id", "split"]]
    out_df.to_csv(OUT_CSV, index=False)

    print(f"\nTotal clips: {len(df)}")
    print(f"Total distinct source videos: {n}")
    print(df["split"].value_counts())
    print(f"\nSaved to: {OUT_CSV}")


if __name__ == "__main__":
    main()