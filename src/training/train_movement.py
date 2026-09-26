"""
src/training/train_movement.py

Trains the temporal camera-movement classifier (LSTM or Transformer head)
on top of frozen CLIP frame embeddings, and reports val macro-average-precision
each epoch -- used to decide between the two temporal_type options honestly,
by measured result rather than guesswork (same spirit as the
class-weighting experiment in PROJECT_NOTES.md section 11).

Macro AP (average precision) is used instead of flat-0.5 macro-F1 for
epoch-to-epoch monitoring because it's threshold-independent -- a model can
have real, useful signal well before its probabilities cross a naive 0.5
cutoff, which is exactly the pitfall documented for the static classifier's
rare classes in section 11. Proper per-class threshold tuning (mirroring
find_best_thresholds.py) should be done separately once an architecture is
chosen.

Usage:
    python -m src.training.train_movement --temporal_type lstm --epochs 5
    python -m src.training.train_movement --temporal_type transformer --epochs 5
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score

from src.data.movement_dataset import MovementDataset
from src.data.movement_label_encoding import MOVEMENT_CLASS_VOCAB
from src.models.movement_classifier import MovementClassifier, DEFAULT_CLIP_CHECKPOINT
from src.models.static_classifier import ClipPreprocess

CHECKPOINT_DIR = Path("checkpoints")


def evaluate(model, loader, device):
    """Returns macro-averaged Average Precision (AP) -- threshold-independent,
    so it's a fair mid-training signal even before per-class thresholds are
    tuned (flat 0.5 macro-F1 undersells the model early on, same issue as
    documented in PROJECT_NOTES.md section 11)."""
    model.eval()
    all_logits = []
    all_labels = []
    with torch.no_grad():
        for frames, labels in loader:
            frames = frames.to(device)
            logits = model(frames)
            all_logits.append(logits.cpu())
            all_labels.append(labels)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    probs = torch.sigmoid(all_logits).numpy()
    labels_np = all_labels.numpy()

    # Per-class AP, skipping classes with zero positive examples in this split
    # (average_precision_score is undefined for an all-zero column)
    ap_scores = []
    for c in range(labels_np.shape[1]):
        if labels_np[:, c].sum() > 0:
            ap_scores.append(average_precision_score(labels_np[:, c], probs[:, c]))
    macro_ap = sum(ap_scores) / len(ap_scores) if ap_scores else 0.0
    return macro_ap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--temporal_type", choices=["lstm", "transformer"], required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    preprocess = ClipPreprocess(DEFAULT_CLIP_CHECKPOINT)

    train_ds = MovementDataset(split="train", transform=preprocess)
    val_ds = MovementDataset(split="val", transform=preprocess)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers,
    )

    num_classes = len(MOVEMENT_CLASS_VOCAB)
    model = MovementClassifier(num_classes=num_classes, temporal_type=args.temporal_type).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(trainable_params, lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for frames, labels in train_loader:
            frames = frames.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(frames)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * frames.size(0)

        avg_train_loss = total_loss / len(train_ds)
        val_ap = evaluate(model, val_loader, device)
        print(f"[{args.temporal_type}] epoch {epoch}/{args.epochs} "
              f"train_loss={avg_train_loss:.4f}  val_macro_ap={val_ap:.4f}")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    ckpt_path = CHECKPOINT_DIR / f"movement_{args.temporal_type}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\nSaved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()