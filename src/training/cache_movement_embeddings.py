"""
src/training/cache_movement_embeddings.py

Runs the frozen CLIP image encoder ONCE over every movement clip (train, val,
test) and saves the resulting frame embeddings to disk, so the temporal head
can be trained and compared in seconds instead of re-running CLIP every epoch.

This is exactly equivalent to what MovementClassifier computes during training,
because CLIP is frozen and there is no data augmentation: the same frames always
give the same embeddings. It uses MovementClassifier._embed_frames() itself, so
the numbers match the current model's.

Output: data/processed/movement_embeddings.pt  (inside data/processed/, which is
gitignored, so it will not be committed).

Usage:
    python -m src.training.cache_movement_embeddings
"""

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.movement_dataset import MovementDataset
from src.data.movement_label_encoding import MOVEMENT_CLASS_VOCAB
from src.models.movement_classifier import DEFAULT_CLIP_CHECKPOINT, MovementClassifier
from src.models.static_classifier import ClipPreprocess

OUT_PATH = Path("data/processed/movement_embeddings.pt")
SPLITS = ("train", "val", "test")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    preprocess = ClipPreprocess(DEFAULT_CLIP_CHECKPOINT)
    # Only the frozen CLIP part is used here; the temporal head is not.
    model = MovementClassifier(
        num_classes=len(MOVEMENT_CLASS_VOCAB), temporal_type="lstm"
    ).to(device)
    model.eval()

    cache = {"classes": list(MOVEMENT_CLASS_VOCAB)}

    for split in SPLITS:
        ds = MovementDataset(split=split, transform=preprocess)
        loader = DataLoader(
            ds, batch_size=args.batch_size, shuffle=False,
            num_workers=args.num_workers,
        )
        all_embeds, all_labels = [], []
        print(f"\n{split}: {len(ds)} clips")
        with torch.no_grad():
            for i, (frames, labels) in enumerate(loader, start=1):
                embeds = model._embed_frames(frames.to(device))
                all_embeds.append(embeds.cpu())
                all_labels.append(labels)
                if i % 50 == 0 or i == len(loader):
                    print(f"  batch {i} of {len(loader)}")

        embeds = torch.cat(all_embeds)
        labels = torch.cat(all_labels)
        assert embeds.shape[0] == len(ds), (embeds.shape, len(ds))
        assert embeds.shape[1] == 16 and embeds.shape[2] == 512, embeds.shape
        cache[split] = {
            "filenames": list(ds.filenames),
            "embeds": embeds,
            "labels": labels,
        }
        print(f"  saved shapes: embeds {tuple(embeds.shape)}, labels {tuple(labels.shape)}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, OUT_PATH)
    size_mb = OUT_PATH.stat().st_size / 1e6
    print(f"\nSaved {OUT_PATH} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()