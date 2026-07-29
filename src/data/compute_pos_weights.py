"""
Compute per-class pos_weight values for BCEWithLogitsLoss, using the
train split's label frequencies. Standard formula per class:

    pos_weight = num_negatives / num_positives

This tells the loss function to penalize a missed positive on a rare
class more heavily than a missed positive on a common class -- helps
counter the "recall stuck near 0 for rare classes" pattern seen in
eval_metrics.csv.

Reuses ShotDataset's already-loaded img_ids/meta_by_id directly (no
image loading needed here), so this runs in seconds.

Usage:
    python -m src.data.compute_pos_weights
"""
import json
from pathlib import Path

import torch

from src.data.dataset import ShotDataset
from src.data.label_encoding import CLASS_VOCAB, SOURCE_FIELD, encode_multihot

MAX_POS_WEIGHT = 20.0  # cap to avoid unstable training / over-predicting rare classes

def main():
    print("Loading train split (labels only, no images)...")
    ds = ShotDataset(split="train", transform=None)
    print(f"train split size: {len(ds)}")

    dimensions = list(CLASS_VOCAB.keys())
    pos_counts = {dim: [0] * len(CLASS_VOCAB[dim]) for dim in dimensions}
    n_samples = len(ds.img_ids)

    for img_id in ds.img_ids:
        meta = ds.meta_by_id[img_id]
        for dim in dimensions:
            source_field = SOURCE_FIELD[dim]
            raw_value = meta.get(source_field)
            vec = encode_multihot(raw_value, dim)
            for i, v in enumerate(vec):
                pos_counts[dim][i] += v

    pos_weights = {}
    print("\n" + "=" * 70)
    for dim in dimensions:
        class_names = CLASS_VOCAB[dim]
        weights = []
        print(f"\n{dim}:")
        for i, cls_name in enumerate(class_names):
            num_pos = pos_counts[dim][i]
            num_neg = n_samples - num_pos
            # Guard against div-by-zero if a class never appears in train
            raw_weight = (num_neg / num_pos) if num_pos > 0 else 1.0
            weight = min(raw_weight, MAX_POS_WEIGHT)
            weights.append(weight)
            print(f"  {cls_name:25s}  pos={num_pos:6d}  neg={num_neg:6d}  pos_weight={weight:.2f}")
        pos_weights[dim] = weights

    print("=" * 70)

    output_path = Path("reports/pos_weights.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(pos_weights, f, indent=2)
    print(f"\nSaved pos_weights to: {output_path}")


if __name__ == "__main__":
    main()