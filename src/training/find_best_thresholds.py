"""
Sweep per-class thresholds on the val split to find the threshold that
maximizes F1 for EACH class individually, instead of using one flat
0.5 for everything (see PROJECT_NOTES.md section 2b -- "per-class
threshold at inference" was always the plan for multi-label).

Saves the resulting thresholds to a JSON file, then re-runs evaluation
using those tuned thresholds so you can see the improvement directly.

Usage:
    python -m src.training.find_best_thresholds --checkpoint checkpoints/best_model.pt
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from src.data.dataset import ShotDataset
from src.data.label_encoding import CLASS_VOCAB
from src.models.static_classifier import StaticShotClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--output-json", type=str, default="reports/best_thresholds.json")
    parser.add_argument("--output-csv", type=str, default="reports/eval_metrics_tuned.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading model...")
    model = StaticShotClassifier(class_vocab=CLASS_VOCAB)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print("Loading val split...")
    transform = model.get_preprocess()
    dataset = ShotDataset(split="val", transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers)
    print(f"val split size: {len(dataset)}")

    dimensions = list(CLASS_VOCAB.keys())
    all_probs = {dim: [] for dim in dimensions}
    all_targets = {dim: [] for dim in dimensions}

    # --- Single forward pass over val set, collect raw probabilities ---
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            logits = model(images)
            for dim in dimensions:
                probs = torch.sigmoid(logits[dim]).cpu()
                all_probs[dim].append(probs)
                all_targets[dim].append(labels[dim].int())
            if batch_idx % 20 == 0:
                print(f"  batch {batch_idx}/{len(loader)}")

    # --- Sweep thresholds per class ---
    threshold_grid = np.arange(0.05, 0.95, 0.05)
    best_thresholds = {}  # {dim: [threshold_per_class]}
    rows = []

    print("\n" + "=" * 70)
    for dim in dimensions:
        probs = torch.cat(all_probs[dim], dim=0).numpy()  # (N, n_classes)
        targets = torch.cat(all_targets[dim], dim=0).numpy()
        class_names = CLASS_VOCAB[dim]
        n_classes = len(class_names)

        dim_thresholds = []
        print(f"\n{dim}:")
        for i, cls_name in enumerate(class_names):
            best_f1 = -1.0
            best_t = 0.5
            for t in threshold_grid:
                preds = (probs[:, i] >= t).astype(int)
                _, _, f1, _ = precision_recall_fscore_support(
                    targets[:, i], preds, average="binary", zero_division=0
                )
                if f1 > best_f1:
                    best_f1 = f1
                    best_t = float(t)
            dim_thresholds.append(best_t)

            # Recompute final precision/recall/support at the chosen threshold
            final_preds = (probs[:, i] >= best_t).astype(int)
            p, r, f1, support = precision_recall_fscore_support(
                targets[:, i], final_preds, average="binary", zero_division=0
            )
            print(f"  {cls_name:25s}  threshold={best_t:.2f}  "
                  f"P={p:.3f}  R={r:.3f}  F1={f1:.3f}  support={int(targets[:, i].sum())}")
            rows.append({
                "dimension": dim, "class": cls_name, "threshold": best_t,
                "precision": p, "recall": r, "f1": f1,
            })

        best_thresholds[dim] = dim_thresholds
        macro_f1 = np.mean([row["f1"] for row in rows if row["dimension"] == dim])
        print(f"  --> {dim} macro-F1 (tuned): {macro_f1:.3f}")

    print("=" * 70)

    # --- Save thresholds JSON ---
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(best_thresholds, f, indent=2)
    print(f"\nSaved per-class thresholds to: {args.output_json}")

    # --- Save detailed CSV ---
    import csv
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dimension", "class", "threshold", "precision", "recall", "f1"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved tuned metrics to: {args.output_csv}")


if __name__ == "__main__":
    main()