"""
src/training/movement_prior_baseline.py

A "no model" baseline for the movement classifier. It ignores the video
completely: it ranks the classes by how common they are in the TRAIN clips and
gives every val clip the same ranking.

This shows how much of the trained model's val score could be reached just by
knowing which classes are common. It uses the cached labels from
data/processed/movement_embeddings.pt (no GPU, no CLIP, runs in seconds).

Reports the same numbers as train_movement_cached.py on the val split:
top-1 hit rate, top-2 hit rate, and macro average precision (AP).
With one constant score per class, AP for a class equals how common that
class is in val.

Usage:
    python -m src.training.movement_prior_baseline
"""

from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

CACHE_PATH = Path("data/processed/movement_embeddings.pt")


def main():
    if not CACHE_PATH.exists():
        raise SystemExit(f"{CACHE_PATH} not found. Run: "
                         "python -m src.training.cache_movement_embeddings")

    cache = torch.load(CACHE_PATH, weights_only=True)
    classes = cache["classes"]
    y_train = cache["train"]["labels"]
    y_val = cache["val"]["labels"]

    train_freq = y_train.mean(dim=0)  # fraction of train clips with each class
    ranking = train_freq.argsort(descending=True)
    top1, top2 = ranking[:1], ranking[:2]

    hit1 = (y_val[:, top1].sum(dim=1) > 0).float().mean().item()
    hit2 = (y_val[:, top2].sum(dim=1) > 0).float().mean().item()

    aps = []
    for c in range(y_val.shape[1]):
        if y_val[:, c].sum() > 0:
            scores = np.full(y_val.shape[0], train_freq[c].item())
            aps.append(average_precision_score(y_val[:, c].numpy(), scores))
    macro_ap = sum(aps) / len(aps)

    print("Most common classes in TRAIN (fraction of clips):")
    for i in ranking[:5].tolist():
        print(f"  {classes[i]}: {train_freq[i].item():.3f}")
    print()
    print(f"Always predict top-1 = {classes[top1.item()]}")
    print(f"Always predict top-2 = {[classes[i] for i in top2.tolist()]}")
    print()
    print(f"PRIOR BASELINE (val): top1={hit1:.3f} top2={hit2:.3f} "
          f"macro_ap={macro_ap:.4f}  ({len(aps)} classes with val support)")


if __name__ == "__main__":
    main()