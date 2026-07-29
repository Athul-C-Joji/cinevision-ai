"""
smoke_test.py — End-to-end check: real data -> model -> loss

Run this BEFORE writing the full training loop. It catches:
  - Dataset <-> model transform mismatches
  - Shape mismatches between labels and model outputs
  - Whether BCEWithLogitsLoss actually runs cleanly across all 7 heads
  - Whether this fits on the GTX 1650 without an out-of-memory error

If this script runs cleanly to the end, train.py is safe to write next.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import ShotDataset
from src.data.label_encoding import CLASS_VOCAB
from src.models.static_classifier import StaticShotClassifier


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # --- Build model first (so we can get its matching preprocess fn) ---
    print("\nLoading model...")
    model = StaticShotClassifier(CLASS_VOCAB)
    model.to(device)
    transform = model.get_preprocess()

    # --- Build a small dataset + loader ---
    print("\nBuilding train dataset (this loads meta.jsonl into memory)...")
    train_ds = ShotDataset(split="train", transform=transform)
    print(f"Train dataset size: {len(train_ds)}")

    # Small batch size to start -- GTX 1650 has 4GB VRAM. We'll only run
    # ONE batch here as a smoke test, so batch size just needs to be
    # small enough to comfortably fit, not tuned for real training yet.
    BATCH_SIZE = 8
    loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,  # start at 0 on Windows -- multiprocessing workers
                         # can be flaky on Windows/PowerShell; bump later
                         # once this baseline works, if you want faster
                         # data loading.
    )

    # --- Pull ONE real batch ---
    print(f"\nFetching one batch (batch_size={BATCH_SIZE})...")
    images, labels = next(iter(loader))
    print(f"images.shape: {images.shape}  (expect: [{BATCH_SIZE}, 3, 224, 224])")
    print("labels shapes:")
    for dim, tensor in labels.items():
        print(f"  {dim}: {tuple(tensor.shape)}")

    images = images.to(device)
    labels = {dim: t.to(device) for dim, t in labels.items()}

    # --- Forward pass ---
    print("\nRunning forward pass...")
    model.eval()  # heads have no dropout/batchnorm yet, but good habit
    outputs = model(images)
    print("Output logits shapes:")
    for dim, logits in outputs.items():
        print(f"  {dim}: {tuple(logits.shape)}")

    # --- Loss computation, per head, matching the multi-label decision ---
    print("\nComputing BCEWithLogitsLoss per head...")
    criterion = nn.BCEWithLogitsLoss()
    losses = {}
    total_loss = 0.0
    for dim in CLASS_VOCAB.keys():
        loss = criterion(outputs[dim], labels[dim])
        losses[dim] = loss.item()
        total_loss += loss

    for dim, val in losses.items():
        print(f"  {dim}: {val:.4f}")
    print(f"\nTotal loss (sum across 7 heads): {total_loss.item():.4f}")

    # --- Confirm gradients only flow into head params, not CLIP ---
    print("\nChecking backward pass (gradients should only touch head params)...")
    total_loss.backward()
    clip_has_grad = any(p.grad is not None for p in model.clip.parameters())
    head_has_grad = any(p.grad is not None for p in model.heads.parameters())
    print(f"  CLIP backbone has gradients: {clip_has_grad}  (expect: False)")
    print(f"  Heads have gradients:        {head_has_grad}  (expect: True)")

    if device == "cuda":
        mem_allocated = torch.cuda.memory_allocated() / 1e9
        mem_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"\nGPU memory allocated: {mem_allocated:.2f} GB")
        print(f"GPU memory reserved:  {mem_reserved:.2f} GB")
        print("(GTX 1650 has 4GB total -- if this is already close to 4GB "
              "with batch_size=8, we'll need to reduce batch size or use "
              "mixed precision for real training)")

    print("\n✅ Smoke test passed. Safe to write train.py next.")


if __name__ == "__main__":
    main()