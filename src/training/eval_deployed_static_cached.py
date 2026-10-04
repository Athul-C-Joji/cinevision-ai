"""
src/training/eval_deployed_static_cached.py

Scores the DEPLOYED static model (checkpoints/best_model.pt) on the cached
embeddings, with its saved thresholds (reports/best_thresholds.json), as a
reproduction check: on val the numbers should match the section 11 table in
PROJECT_NOTES.md (tuned macro-F1 Frame Size 0.651, Lens Size 0.505,
Composition 0.476, Shot Framing 0.660, Camera Angle 0.568, Lighting Type 0.423,
Lighting 0.499) up to small differences.

The thresholds were tuned on val, so val numbers are optimistic. With
--eval-test the same val-tuned thresholds are applied to the test split: that
is the honest baseline for comparing new heads. Use --eval-test sparingly.

Nothing is modified or saved.

Usage:
    python -m src.training.eval_deployed_static_cached
    python -m src.training.eval_deployed_static_cached --eval-test
"""

import argparse
import json
from pathlib import Path

import torch

from src.training.static_cached_common import (
    DEFAULT_CACHE, StaticHeads, collect, load_cache, print_scores, score_split,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--thresholds", type=Path, default=Path("reports/best_thresholds.json"))
    parser.add_argument("--eval-test", action="store_true")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = load_cache(args.cache)
    dims = cache["dimensions"]

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    heads_state = {k: v for k, v in ckpt["model_state_dict"].items() if k.startswith("heads.")}
    model = StaticHeads(cache["val"]["embeds"].shape[1], cache["class_vocab"], "linear")
    model.load_state_dict(heads_state, strict=True)  # errors on any name/shape mismatch
    model.to(device)
    print(f"Loaded {len(heads_state)} head tensors from {args.checkpoint}")

    with open(args.thresholds) as f:
        thresholds = json.load(f)

    probs, targets, loss = collect(model, cache["val"], device, dims)
    ap, f1 = score_split(probs, targets, thresholds, dims)
    print(f"\nVal images: {cache['val']['embeds'].shape[0]} "
          f"(unreadable skipped: {cache['val']['n_unreadable']})")
    print(f"Val loss (sum over 7 heads): {loss:.4f}")
    print_scores("DEPLOYED MODEL, val, saved thresholds (tuned on val: optimistic)",
                 ap, f1, dims, targets)

    if args.eval_test:
        probs, targets, loss = collect(model, cache["test"], device, dims)
        ap, f1 = score_split(probs, targets, thresholds, dims)
        print(f"\nTest images: {cache['test']['embeds'].shape[0]} "
              f"(unreadable skipped: {cache['test']['n_unreadable']})")
        print(f"Test loss (sum over 7 heads): {loss:.4f}")
        print_scores("DEPLOYED MODEL, test, val-tuned thresholds", ap, f1, dims, targets)


if __name__ == "__main__":
    main()