"""
src/training/train_static_cached.py

Trains the 7 static heads on CACHED CLIP embeddings (made by
cache_static_embeddings.py). CLIP itself is not run here, so a run takes
minutes, not hours.

Model choice uses the VAL split only:
  - the best epoch is the one with the highest mean val macro-AP
    (threshold-free);
  - per-class thresholds are then tuned on val (same method as
    find_best_thresholds.py), so the val F1 numbers are optimistic, exactly
    like the deployed model's val numbers.

The TEST split is only touched with --eval-test (use it once, for the final
candidate): val-tuned thresholds are applied to test.

Nothing in checkpoints/best_model.pt or reports/ is modified. Results are
saved to checkpoints/experiments/ (gitignored). The saved file name contains
the backbone name taken from the cache, so runs on different caches do not
overwrite each other.

Usage:
    python -m src.training.train_static_cached --head linear --epochs 5 --batch-size 32 --seed 1
    python -m src.training.train_static_cached --head mlp --epochs 10 --lr 0.0003 --dropout 0.5
    python -m src.training.train_static_cached --head mlp --epochs 10 --lr 0.0003 --dropout 0.5 --cache data/processed/static_embeddings_clip-vit-base-patch16.pt
"""

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from src.training.static_cached_common import (
    DEFAULT_CACHE, StaticHeads, collect, load_cache, macro_ap, print_scores,
    score_split, tune_thresholds,
)

# Reference numbers pasted from eval_deployed_static_cached.py (val, deployed model,
# which uses CLIP ViT-B/32).
DEPLOYED_VAL_LOSS = 1.8997
DEPLOYED_VAL_MEAN_AP = 0.532
DEPLOYED_VAL_MEAN_F1 = 0.540


def to_device(split, dims, device):
    x = split["embeds"].float().to(device)
    labels = {d: split["labels"][d].float().to(device) for d in dims}
    return x, labels


def train_one_epoch(model, x, labels, dims, optimizer, batch_size, gen):
    model.train()
    n = x.shape[0]
    perm = torch.randperm(n, generator=gen).to(x.device)
    total = 0.0
    for i in range(0, n, batch_size):
        idx = perm[i:i + batch_size]
        out = model(x[idx])
        loss = sum(F.binary_cross_entropy_with_logits(out[d], labels[d][idx]) for d in dims)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total += loss.item() * len(idx)
    return total / n


def mean_val_ap(model, val, device, dims):
    probs, targets, loss = collect(model, val, device, dims)
    aps = [macro_ap(probs[d], targets[d]) for d in dims]
    return float(np.mean(aps)), loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--head", choices=["linear", "mlp"], default="linear")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--hidden", type=int, default=512, help="MLP hidden size")
    parser.add_argument("--dropout", type=float, default=0.2, help="MLP dropout")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--eval-test", action="store_true",
                        help="also score the TEST split with val-tuned thresholds (use once)")
    parser.add_argument("--out-dir", type=Path, default=Path("checkpoints/experiments"))
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    cache = load_cache(args.cache)
    dims = cache["dimensions"]
    in_dim = cache["train"]["embeds"].shape[1]
    backbone = str(cache.get("clip_checkpoint", "unknown")).split("/")[-1]

    x_train, y_train = to_device(cache["train"], dims, device)
    print(f"Backbone (from cache): {backbone}")
    print(f"Train {x_train.shape[0]} | val {cache['val']['embeds'].shape[0]} | "
          f"test {cache['test']['embeds'].shape[0]} | embedding size {in_dim}")

    model = StaticHeads(in_dim, cache["class_vocab"], args.head,
                        hidden=args.hidden, dropout=args.dropout).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Head: {args.head} | trainable params: {n_params:,} | lr {args.lr} | "
          f"weight decay {args.weight_decay} | batch {args.batch_size} | seed {args.seed}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    best_ap, best_epoch, best_state = -1.0, 0, None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, x_train, y_train, dims, optimizer,
                                     args.batch_size, gen)
        val_ap, val_loss = mean_val_ap(model, cache["val"], device, dims)
        marker = ""
        if val_ap > best_ap:
            best_ap, best_epoch = val_ap, epoch
            best_state = copy.deepcopy(model.state_dict())
            marker = "  <- best so far"
        print(f"epoch {epoch:>2}/{args.epochs}  train loss {train_loss:.4f}  "
              f"val loss {val_loss:.4f}  val mean macro-AP {val_ap:.4f}  "
              f"({time.time() - t0:.0f}s){marker}")

    model.load_state_dict(best_state)
    print(f"\nBest epoch by val mean macro-AP: {best_epoch} (AP {best_ap:.4f})")

    probs, targets, val_loss = collect(model, cache["val"], device, dims)
    thresholds = tune_thresholds(probs, targets, dims)
    ap, f1 = score_split(probs, targets, thresholds, dims)
    print(f"Val loss at best epoch: {val_loss:.4f}  (deployed model: {DEPLOYED_VAL_LOSS})")
    print_scores(f"THIS RUN ({backbone}), val, thresholds tuned on val (optimistic)",
                 ap, f1, dims, targets)
    print(f"\n  Deployed model reference (val, ViT-B/32 linear): mean macro-AP "
          f"{DEPLOYED_VAL_MEAN_AP}, mean macro-F1 {DEPLOYED_VAL_MEAN_F1}")

    test_scores = None
    if args.eval_test:
        t_probs, t_targets, t_loss = collect(model, cache["test"], device, dims)
        t_ap, t_f1 = score_split(t_probs, t_targets, thresholds, dims)
        print(f"\nTest loss: {t_loss:.4f}")
        print_scores(f"THIS RUN ({backbone}), test, val-tuned thresholds",
                     t_ap, t_f1, dims, t_targets)
        test_scores = {"macro_ap": t_ap, "macro_f1": t_f1, "loss": t_loss}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    tag = (f"static_{backbone}_{args.head}_seed{args.seed}_ep{args.epochs}"
           f"_lr{args.lr:g}_wd{args.weight_decay:g}")
    if args.head == "mlp":
        tag += f"_do{args.dropout:g}_h{args.hidden}"
    out_path = args.out_dir / f"{tag}.pt"
    torch.save({
        "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
        "thresholds": thresholds,
        "best_epoch": best_epoch,
        "clip_checkpoint": cache.get("clip_checkpoint"),
        "val_macro_ap": ap,
        "val_macro_f1": f1,
        "test_scores": test_scores,
        "args": {k: str(v) for k, v in vars(args).items()},
    }, out_path)
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()