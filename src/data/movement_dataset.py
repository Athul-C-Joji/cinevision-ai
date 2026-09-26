"""
src/data/movement_dataset.py

PyTorch Dataset for the temporal camera-movement classifier. Mirrors the
pattern in dataset.py (ShotDataset), but each item is a sequence of frames
(one video clip) rather than a single image.

Wires together:
  - data/processed/movement_splits.csv          (filename -> split, leak-safe by source video)
  - data/processed/movement_labels_encoded.csv  (filename -> multi-hot movement labels)
  - data/processed/movement_frames/<clip_id>/frame_XX.jpg  (16 extracted frames per clip)
  - src/data/movement_label_encoding.py         (MOVEMENT_CLASS_VOCAB)
  - src/models/static_classifier.py             (ClipPreprocess -- reused as-is)

Usage:
    from src.data.movement_dataset import MovementDataset
    from src.models.static_classifier import ClipPreprocess

    preprocess = ClipPreprocess(DEFAULT_CLIP_CHECKPOINT)
    train_ds = MovementDataset(split="train", transform=preprocess)
    frames, label = train_ds[0]
    # frames: tensor of shape (16, 3, H, W)
    # label: tensor of shape (len(MOVEMENT_CLASS_VOCAB),)
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from PIL import Image
import pandas as pd

from src.data.movement_label_encoding import MOVEMENT_CLASS_VOCAB

N_FRAMES = 16

DEFAULT_SPLITS_CSV = Path("data/processed/movement_splits.csv")
DEFAULT_LABELS_CSV = Path("data/processed/movement_labels_encoded.csv")
DEFAULT_FRAMES_DIR = Path("data/processed/movement_frames")


class MovementDataset(Dataset):
    def __init__(
        self,
        split: str,
        transform=None,
        splits_csv: Path = DEFAULT_SPLITS_CSV,
        labels_csv: Path = DEFAULT_LABELS_CSV,
        frames_dir: Path = DEFAULT_FRAMES_DIR,
        n_frames: int = N_FRAMES,
    ):
        """
        Args:
            split: one of "train", "val", "test" -- filters movement_splits.csv
            transform: per-frame image transform (e.g. ClipPreprocess from
                static_classifier.py). Applied independently to each of the
                n_frames frames. If None, returns a list of raw PIL Images.
            splits_csv / labels_csv / frames_dir: override paths if needed
            n_frames: expected frames per clip (must match what
                extract_movement_frames.py produced)
        """
        assert split in ("train", "val", "test"), f"Unknown split: {split}"
        self.split = split
        self.transform = transform
        self.frames_dir = frames_dir
        self.n_frames = n_frames

        splits_df = pd.read_csv(splits_csv)
        splits_df = splits_df[splits_df["split"] == split].reset_index(drop=True)
        self.filenames = splits_df["filename"].tolist()

        self.labels_df = pd.read_csv(labels_csv).set_index("filename")

        missing_labels = [f for f in self.filenames if f not in self.labels_df.index]
        if missing_labels:
            raise ValueError(
                f"{len(missing_labels)} clips in split '{split}' have no "
                f"matching row in {labels_csv}. First few: {missing_labels[:5]}"
            )

        missing_frames = [
            f for f in self.filenames
            if not (frames_dir / Path(f).stem).exists()
        ]
        if missing_frames:
            raise ValueError(
                f"{len(missing_frames)} clips in split '{split}' have no "
                f"extracted frames directory under {frames_dir}. "
                f"First few: {missing_frames[:5]}. Did you exclude "
                f"DECODE_FAILED clips when building movement_splits.csv?"
            )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        clip_id = Path(filename).stem
        clip_dir = self.frames_dir / clip_id

        frames = []
        for i in range(self.n_frames):
            frame_path = clip_dir / f"frame_{i:02d}.jpg"
            image = Image.open(frame_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            frames.append(image)

        if self.transform is not None:
            frames = torch.stack(frames, dim=0)  # (n_frames, 3, H, W)

        label_row = self.labels_df.loc[filename, MOVEMENT_CLASS_VOCAB]
        label = torch.tensor(label_row.values.astype("float32"))

        return frames, label


if __name__ == "__main__":
    # Quick smoke test -- run with: python -m src.data.movement_dataset
    ds = MovementDataset(split="train", transform=None)
    print(f"train split size: {len(ds)}")
    frames, label = ds[0]
    print(f"frames type: {type(frames)}, count: {len(frames)}")
    print(f"label: {label.tolist()}")