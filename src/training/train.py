"""
src/training/train.py — Training loop for the 7-head static classifier

Designed to run LOCALLY on the GTX 1650 first for debugging/tuning
(smaller batch size, mixed precision, short runs), then move to Kaggle
for the real full-scale run later. Nothing in here is Kaggle-specific,
so the same script works in both places -- only --epochs / --batch-size
/ --max-samples would typically change between the two.

Usage (local debug run, e.g. 1 epoch on a small subset, just to make
sure everything works before committing to a long run):
    python -m src.training.train --epochs 1 --batch-size 32 --max-samples 500

Usage (a real local run once you trust it):
    python -m src.training.train --epochs 5 --batch-size 32

Outputs:
    checkpoints/best_model.pt   -- best model so far, by val loss
    checkpoints/last_model.pt   -- most recent epoch's model (for resuming)
    reports/training_log.csv    -- per-epoch train/val loss, one row per epoch
"""

import argparse
import csv
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset

from src.data.dataset import ShotDataset
from src.data.label_encoding import CLASS_VOCAB
from src.models.static_classifier import StaticShotClassifier

DIMENSIONS = list(CLASS_VOCAB.keys())

CHECKPOINT_DIR = Path("checkpoints")
REPORTS_DIR = Path("reports")


def compute_multihead_loss(outputs: dict, labels: dict, criterion) -> tuple:
    """
    Sum BCEWithLogitsLoss across all 7 heads. Returns (total_loss, per_head_dict)
    so callers can log both the combined loss (for backward()) and the
    individual head losses (useful for spotting if one dimension is
    struggling much worse than the others).

    `criterion` can be either a single shared BCEWithLogitsLoss (no class
    weighting) or a dict of {dim: BCEWithLogitsLoss(pos_weight=...)} for
    per-class weighted training -- see train.py's setup in main().
    """
    per_head = {}
    total = 0.0
    for dim in DIMENSIONS:
        dim_criterion = criterion[dim] if isinstance(criterion, dict) else criterion
        loss = dim_criterion(outputs[dim], labels[dim])
        per_head[dim] = loss.item()
        total = total + loss  # keep as tensor, not .item(), for backward()
    return total, per_head


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None,
              log_every=50, phase_name="train"):
    """
    One pass over the data. If optimizer is provided, this trains
    (backward + step); otherwise it's a val/eval pass (no gradient updates).
    Returns average total loss and average per-head loss across the epoch.

    Prints progress every `log_every` batches so long epochs never look
    frozen -- with num_workers=0 and real disk I/O, a full epoch can take
    several minutes with otherwise zero console output.
    """
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    running_total = 0.0
    running_per_head = {dim: 0.0 for dim in DIMENSIONS}
    n_batches = 0
    total_batches = len(loader)
    epoch_start = time.time()

    for images, labels in loader:
        images = images.to(device)
        labels = {dim: t.to(device) for dim, t in labels.items()}

        if is_train:
            optimizer.zero_grad()

        # Mixed precision on CUDA -- matches PROJECT_NOTES.md compute
        # strategy (section 6) for making local GTX 1650 training feasible.
        if scaler is not None and is_train:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                total_loss, per_head = compute_multihead_loss(outputs, labels, criterion)
            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            total_loss, per_head = compute_multihead_loss(outputs, labels, criterion)
            if is_train:
                total_loss.backward()
                optimizer.step()

        running_total += total_loss.item()
        for dim in DIMENSIONS:
            running_per_head[dim] += per_head[dim]
        n_batches += 1

        if n_batches % log_every == 0 or n_batches == total_batches:
            elapsed = time.time() - epoch_start
            avg_so_far = running_total / n_batches
            batches_per_sec = n_batches / elapsed if elapsed > 0 else 0
            eta_sec = (total_batches - n_batches) / batches_per_sec if batches_per_sec > 0 else 0
            print(f"    [{phase_name}] batch {n_batches}/{total_batches}  "
                  f"avg_loss={avg_so_far:.4f}  "
                  f"({batches_per_sec:.2f} batch/s, ETA {eta_sec:.0f}s)")

    avg_total = running_total / n_batches
    avg_per_head = {dim: v / n_batches for dim, v in running_per_head.items()}
    return avg_total, avg_per_head


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=3,
                         help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                         help="Batch size. GTX 1650 (4GB) has headroom "
                              "well above 8 based on smoke test -- 32 is "
                              "a reasonable starting point, raise if no "
                              "OOM errors.")
    parser.add_argument("--lr", type=float, default=1e-3,
                         help="Learning rate for the head optimizer")
    parser.add_argument("--max-samples", type=int, default=None,
                         help="If set, only use this many samples from "
                              "train/val (for fast local debug runs). "
                              "Leave unset for a real full run.")
    parser.add_argument("--num-workers", type=int, default=0,
                         help="DataLoader worker processes. Start at 0 on "
                              "Windows (multiprocessing can be flaky in "
                              "PowerShell); raise later if stable.")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    REPORTS_DIR.mkdir(exist_ok=True)

    # --- Model ---
    print("\nLoading model...")
    model = StaticShotClassifier(CLASS_VOCAB)
    model.to(device)
    transform = model.get_preprocess()

    # --- Data ---
    print("\nLoading datasets...")
    train_ds = ShotDataset(split="train", transform=transform)
    val_ds = ShotDataset(split="val", transform=transform)

    if args.max_samples is not None:
        train_ds = Subset(train_ds, range(min(args.max_samples, len(train_ds))))
        val_ds = Subset(val_ds, range(min(args.max_samples, len(val_ds))))
        print(f"DEBUG MODE: capped to {len(train_ds)} train / {len(val_ds)} val samples")
    else:
        print(f"Full dataset: {len(train_ds)} train / {len(val_ds)} val samples")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                               shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=args.num_workers)

    # --- Optimizer, loss, mixed precision ---
    # Only the heads have requires_grad=True, so this optimizer only ever
    # touches those params -- confirmed by the smoke test's gradient check.
    optimizer = torch.optim.Adam(model.trainable_parameters(), lr=args.lr)
    # --- Load per-class pos_weights (see src/data/compute_pos_weights.py) ---
    import json
    pos_weights_path = Path("reports/pos_weights.json")
    if pos_weights_path.exists():
        with open(pos_weights_path) as f:
            pos_weights_raw = json.load(f)
        criterion = {
            dim: nn.BCEWithLogitsLoss(
                pos_weight=torch.tensor(pos_weights_raw[dim], device=device)
            )
            for dim in DIMENSIONS
        }
        print(f"Loaded per-class pos_weights from {pos_weights_path}")
    else:
        print(f"WARNING: {pos_weights_path} not found -- using unweighted "
              f"BCEWithLogitsLoss for all heads. Run "
              f"src/data/compute_pos_weights.py first if you want class weighting.")
        criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler('cuda') if device == "cuda" else None

    # --- CSV log setup ---
    log_path = REPORTS_DIR / "training_log.csv"
    fieldnames = ["epoch", "train_loss", "val_loss"] + \
                 [f"train_{d}" for d in DIMENSIONS] + \
                 [f"val_{d}" for d in DIMENSIONS] + \
                 ["epoch_time_sec"]
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    # --- Training loop ---
    best_val_loss = float("inf")
    print(f"\nStarting training: {args.epochs} epochs, batch_size={args.batch_size}\n")

    for epoch in range(1, args.epochs + 1):
        start = time.time()

        train_loss, train_per_head = run_epoch(
            model, train_loader, criterion, device, optimizer=optimizer,
            scaler=scaler, phase_name="train"
        )
        val_loss, val_per_head = run_epoch(
            model, val_loader, criterion, device, optimizer=None,
            phase_name="val"
        )

        elapsed = time.time() - start

        print(f"Epoch {epoch}/{args.epochs}  "
              f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
              f"({elapsed:.1f}s)")

        # Log to CSV
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "epoch_time_sec": round(elapsed, 1),
        }
        for d in DIMENSIONS:
            row[f"train_{d}"] = train_per_head[d]
            row[f"val_{d}"] = val_per_head[d]
        with open(log_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)

        # Save "last" checkpoint every epoch (lets you resume if interrupted)
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
        }, CHECKPOINT_DIR / "last_model.pt")

        # Save "best" checkpoint only when val loss improves
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
            }, CHECKPOINT_DIR / "best_model.pt")
            print(f"  -> New best val_loss ({val_loss:.4f}), saved best_model.pt")

    print(f"\nDone. Best val_loss: {best_val_loss:.4f}")
    print(f"Log saved to: {log_path}")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}/")


if __name__ == "__main__":
    main()