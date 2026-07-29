"""
src/data/dataset.py — PyTorch Dataset for the 6-head multi-label classifier.

Wires together:
  - data/processed/film_splits.csv  (img_id -> split, leak-safe by film)
  - data/raw/meta.jsonl               (img_id -> raw label strings + metadata)
  - data/raw/images/<img_id>          (flat structure, confirmed 100% coverage)
  - src/data/label_encoding.py        (CLASS_VOCAB + encode_multihot)

Usage:
    from src.data.dataset import ShotDataset
    from src.data.label_encoding import CLASS_VOCAB

    train_ds = ShotDataset(split="train", transform=clip_preprocess)
    img, labels = train_ds[0]
    # labels is a dict: {"Frame Size": tensor([...]), "Shot Type": tensor([...]), ...}
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

from src.data.label_encoding import CLASS_VOCAB, SOURCE_FIELD, encode_multihot

# 7 static-frame dimensions (camera movement is handled separately by the
# temporal model). "Shot Framing" and "Camera Angle" both read from the
# same raw "Shot Type" field in meta.jsonl -- SOURCE_FIELD maps each
# dimension name to the actual meta.jsonl key it should read from.

DIMENSIONS = list(CLASS_VOCAB.keys())

DEFAULT_SPLITS_CSV = Path("data/processed/film_splits.csv")
DEFAULT_META_JSONL = Path("data/raw/meta.jsonl")
DEFAULT_IMAGES_DIR = Path("data/raw/images")
DEFAULT_BROKEN_IMAGES_CSV = Path("reports/broken_images.csv")


class ShotDataset(Dataset):
    def __init__(
        self,
        split: str,
        transform=None,
        splits_csv: Path = DEFAULT_SPLITS_CSV,
        meta_jsonl: Path = DEFAULT_META_JSONL,
        images_dir: Path = DEFAULT_IMAGES_DIR,
        broken_images_csv: Path = DEFAULT_BROKEN_IMAGES_CSV,
    ):
        """
        Args:
            split: one of "train", "val", "test" — filters film_splits.csv
            transform: image transform to apply (e.g. CLIP's preprocess
                fn). If None, returns a raw PIL Image — fine for
                inspection, but you'll want to pass CLIP's preprocess
                for actual training so normalization/resizing matches
                the frozen backbone's expected input.
            splits_csv / meta_jsonl / images_dir: override paths if needed
            broken_images_csv: output of validate_images.py -- a CSV of
                img_ids that failed to load (corrupted, truncated, or
                not actually valid images despite the .jpg extension).
                If this file exists, those img_ids are excluded here so
                training never crashes on them. If it doesn't exist yet
                (validate_images.py hasn't been run), no filtering happens
                and you may hit PIL errors during training -- run
                validate_images.py first if that happens.
        """
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split
        self.transform = transform
        self.images_dir = images_dir

        # --- Load split assignment, filter to this split ---
        splits_df = pd.read_csv(splits_csv)
        splits_df = splits_df[splits_df["split"] == split].reset_index(drop=True)
        self.img_ids = splits_df["img_id"].tolist()

        # --- Exclude known-broken images, if validate_images.py has been run ---
        if broken_images_csv.exists():
            broken_df = pd.read_csv(broken_images_csv)
            broken_ids = set(broken_df["img_id"])
            before = len(self.img_ids)
            self.img_ids = [i for i in self.img_ids if i not in broken_ids]
            excluded = before - len(self.img_ids)
            if excluded > 0:
                print(f"[{split}] Excluded {excluded} known-broken images "
                      f"(from {broken_images_csv})")
        else:
            print(f"[{split}] WARNING: {broken_images_csv} not found -- "
                  f"broken images will NOT be filtered. Run "
                  f"validate_images.py first to avoid crashes.")

        valid_ids = set(self.img_ids)

        # --- Load meta.jsonl, keep only rows in this split ---
        # meta.jsonl has 61,405 rows but some img_ids repeat (multiple
        # QA rows per image). We index by img_id -> raw label dict here;
        # if an img_id appears more than once, later rows overwrite
        # earlier ones for label fields. This assumes label fields are
        # consistent across repeated rows for the same img_id (true for
        # ShotDeck-sourced metadata, which is per-image, not per-QA-pair).
        self.meta_by_id = {}
        with open(meta_jsonl, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                img_id = row["img_id"]
                if img_id in valid_ids:
                    self.meta_by_id[img_id] = row

        # Sanity check: every img_id in this split should have metadata
        missing = valid_ids - set(self.meta_by_id.keys())
        if missing:
            raise ValueError(
                f"{len(missing)} img_ids in split '{split}' have no matching "
                f"row in meta.jsonl. First few: {list(missing)[:5]}"
            )

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        meta = self.meta_by_id[img_id]

        # --- Load image ---
        img_path = self.images_dir / img_id
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # --- Encode all 6 label dimensions as multi-hot vectors ---
        labels = {}
        for dim in DIMENSIONS:
            source_field = SOURCE_FIELD[dim]
            raw_value = meta.get(source_field)  # e.g. "2 shot, High angle, Overhead"
            vec = encode_multihot(raw_value, dim)
            labels[dim] = torch.tensor(vec, dtype=torch.float32)

        return image, labels


if __name__ == "__main__":
    # Quick smoke test — run with: python -m src.data.dataset
    ds = ShotDataset(split="train", transform=None)
    print(f"train split size: {len(ds)}")
    img, labels = ds[0]
    print(f"image type: {type(img)}, size: {img.size}")
    for dim, vec in labels.items():
        print(f"  {dim}: {vec.tolist()}")