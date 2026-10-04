"""
src/training/eval_deployed_movement.py

Measures the DEPLOYED movement model (checkpoints/movement_lstm.pt, the one the
app uses) on the val split, with exactly the same metrics as
train_movement_cached.py: macro average precision (AP), top-1 hit rate, top-2
hit rate. That makes it comparable with the cached-embedding experiment and
with the no-model prior baseline.

How: the checkpoint holds frozen CLIP weights plus the trained LSTM head
('temporal.*' and 'classifier.*' keys). CLIP is frozen and the same for every
run, so this script loads only the head and scores it on the cached val
embeddings (data/processed/movement_embeddings.pt). Loading is strict, so a
shape mismatch stops the script with an error instead of giving wrong numbers.

Only the val split is used. The test split is NOT touched.
Nothing is modified or saved.

Usage:
    python -m src.training.eval_deployed_movement
    python -m src.training.eval_deployed_movement --checkpoint checkpoints\\movement_lstm_v1.pt
"""

import argparse
from pathlib import Path

import torch

from src.training.train_movement_cached import (
    CACHE_PATH,
    MovementHead,
    build_features,
    evaluate,
)

DEFAULT_CHECKPOINT = Path("checkpoints/movement_lstm.pt")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    args = parser.parse_args()

    if not CACHE_PATH.exists():
        raise SystemExit(f"{CACHE_PATH} not found. Run: "
                         "python -m src.training.cache_movement_embeddings")
    if not args.checkpoint.exists():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(CACHE_PATH, weights_only=True)
    x_val = build_features(cache["val"]["embeds"], "embed")
    y_val = cache["val"]["labels"]

    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    head_state = {k: v for k, v in state.items()
                  if k.startswith("temporal.") or k.startswith("classifier.")}
    if not head_state:
        raise SystemExit("No 'temporal.*' / 'classifier.*' keys found in the checkpoint. "
                         f"First keys: {list(state.keys())[:5]}")

    head = MovementHead(input_dim=x_val.shape[2], num_classes=y_val.shape[1])
    head.load_state_dict(head_state, strict=True)  # errors on any shape/name mismatch
    head.to(device)

    ap, hit1, hit2 = evaluate(head, x_val, y_val, device)

    print(f"Checkpoint: {args.checkpoint}")
    print(f"Loaded {len(head_state)} head tensors; "
          f"ignored {len(state) - len(head_state)} other tensors (frozen CLIP).")
    print(f"Val clips: {x_val.shape[0]}")
    print()
    print(f"DEPLOYED MODEL (val): top1={hit1:.3f} top2={hit2:.3f} macro_ap={ap:.4f}")


if __name__ == "__main__":
    main()