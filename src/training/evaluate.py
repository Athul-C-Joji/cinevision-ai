"""
Evaluate a trained StaticShotClassifier checkpoint on a data split.

Computes per-class precision/recall/F1 for each of the 7 multi-label
heads (NOT a single confusion matrix -- see PROJECT_NOTES.md section 2b
for why: multi-label needs per-class metrics, not single-label accuracy).

Usage:
    python -m src.training.evaluate --checkpoint checkpoints/best_model.pt --split val
    python -m src.training.evaluate --checkpoint checkpoints/best_model.pt --split test
"""
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_fscore_support

from src.data.dataset import ShotDataset
from src.data.label_encoding import CLASS_VOCAB
from src.models.static_classifier import StaticShotClassifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True,
                         help="Path to checkpoint file, e.g. checkpoints/best_model.pt")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5,
                         help="Sigmoid threshold for turning probabilities into 0/1 predictions")
    parser.add_argument("--output-csv", type=str, default="reports/eval_metrics.csv")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load model ---
    print("Loading model...")
    model = StaticShotClassifier(class_vocab=CLASS_VOCAB)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', '?')}, "
          f"val_loss={checkpoint.get('val_loss', '?')}")

    # --- Load dataset ---
    print(f"Loading {args.split} split...")
    transform = model.get_preprocess()
    dataset = ShotDataset(split=args.split, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers)
    print(f"{args.split} split size: {len(dataset)}")

    # --- Collect predictions + ground truth per head ---
    dimensions = list(CLASS_VOCAB.keys())
    all_preds = {dim: [] for dim in dimensions}
    all_targets = {dim: [] for dim in dimensions}

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(loader):
            images = images.to(device)
            logits = model(images)  # dict of {dim: (batch, n_classes)}

            for dim in dimensions:
                probs = torch.sigmoid(logits[dim]).cpu()
                preds = (probs >= args.threshold).int()
                all_preds[dim].append(preds)
                all_targets[dim].append(labels[dim].int())

            if batch_idx % 20 == 0:
                print(f"  batch {batch_idx}/{len(loader)}")

    # --- Compute per-class + per-head metrics ---
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    rows = []

    print("\n" + "=" * 70)
    for dim in dimensions:
        preds = torch.cat(all_preds[dim], dim=0).numpy()
        targets = torch.cat(all_targets[dim], dim=0).numpy()
        class_names = CLASS_VOCAB[dim]

        precision, recall, f1, support = precision_recall_fscore_support(
            targets, preds, average=None, zero_division=0
        )

        print(f"\n{dim}:")
        for i, cls_name in enumerate(class_names):
            print(f"  {cls_name:25s}  P={precision[i]:.3f}  R={recall[i]:.3f}  "
                  f"F1={f1[i]:.3f}  support={int(support[i])}")
            rows.append({
                "dimension": dim, "class": cls_name,
                "precision": precision[i], "recall": recall[i],
                "f1": f1[i], "support": int(support[i]),
            })

        macro_f1 = f1.mean()
        print(f"  --> {dim} macro-F1: {macro_f1:.3f}")
        rows.append({
            "dimension": dim, "class": "MACRO_AVG",
            "precision": precision.mean(), "recall": recall.mean(),
            "f1": macro_f1, "support": int(support.sum()),
        })

    print("=" * 70)

    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["dimension", "class", "precision", "recall", "f1", "support"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved detailed metrics to: {args.output_csv}")


if __name__ == "__main__":
    main()