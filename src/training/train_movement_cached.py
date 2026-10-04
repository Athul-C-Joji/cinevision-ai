"""
src/training/train_movement_cached.py

Trains the movement LSTM head on CACHED CLIP embeddings
(made by src/training/cache_movement_embeddings.py), so each run is fast and
several settings and random seeds can be compared.

Input modes (the experiment):
  embed       the 16 frame embeddings only (the same input as the current model)
  diff        only the change between neighbouring frames' embeddings
  embed_diff  both together
Hypothesis (NOT yet tested): frame-to-frame change carries camera-motion
information that single-frame CLIP embeddings do not.

Reports on the val split: macro average precision (AP), and top-1 / top-2 "hit
rate" = how often the top-1 (or at least one of the top-2) predicted class is a
true label. Top-2 matches how the app shows movement.

The best epoch (by val macro AP) is kept. Choosing the best epoch on val makes
the val number slightly optimistic. The test split is only used when you pass
--eval_test, and should be used ONCE, for the final chosen setting.

Saved weights contain the head only (not CLIP) and are NOT loadable by the
current inference code. They go to checkpoints/movement_cached_<mode>_seed<N>.pt
and never overwrite movement_lstm.pt.

Usage:
    python -m src.training.train_movement_cached --mode embed --seed 1
    python -m src.training.train_movement_cached --mode embed_diff --seed 1 --eval_test
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score

CACHE_PATH = Path("data/processed/movement_embeddings.pt")
CHECKPOINT_DIR = Path("checkpoints")
MODES = ("embed", "diff", "embed_diff")


def build_features(embeds, mode):
    """embeds: (n, 16, 512) -> (n, 16, input_dim)."""
    if mode == "embed":
        return embeds
    diffs = embeds[:, 1:] - embeds[:, :-1]                    # (n, 15, 512)
    diffs = torch.cat([torch.zeros_like(embeds[:, :1]), diffs], dim=1)  # (n, 16, 512)
    if mode == "diff":
        return diffs
    return torch.cat([embeds, diffs], dim=2)                  # embed_diff


class MovementHead(nn.Module):
    """Same shape of head as the current LSTM model, but on cached features."""

    def __init__(self, input_dim, num_classes, hidden_dim=128):
        super().__init__()
        self.temporal = nn.LSTM(
            input_size=input_dim, hidden_size=hidden_dim,
            num_layers=1, batch_first=True,
        )
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        _, (h_n, _) = self.temporal(x)
        return self.classifier(h_n[-1])


def evaluate(model, x, y, device):
    """Returns (macro AP, top-1 hit rate, top-2 hit rate)."""
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(x.to(device))).cpu()
    probs_np, y_np = probs.numpy(), y.numpy()

    aps = []
    for c in range(y_np.shape[1]):
        if y_np[:, c].sum() > 0:  # AP is undefined for a class with no positives
            aps.append(average_precision_score(y_np[:, c], probs_np[:, c]))
    macro_ap = sum(aps) / len(aps) if aps else 0.0

    top2 = probs.topk(2, dim=1).indices
    hit1 = (y.gather(1, top2[:, :1]).squeeze(1) > 0.5).float().mean().item()
    hit2 = (y.gather(1, top2) > 0.5).any(dim=1).float().mean().item()
    return macro_ap, hit1, hit2


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--eval_test", action="store_true",
                        help="Also report the test split. Use once, for the final setting.")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if not CACHE_PATH.exists():
        raise SystemExit(f"{CACHE_PATH} not found. Run: "
                         "python -m src.training.cache_movement_embeddings")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cache = torch.load(CACHE_PATH, weights_only=True)

    x_train = build_features(cache["train"]["embeds"], args.mode).to(device)
    y_train = cache["train"]["labels"].to(device)
    x_val = build_features(cache["val"]["embeds"], args.mode)
    y_val = cache["val"]["labels"]

    num_classes = y_train.shape[1]
    model = MovementHead(x_train.shape[2], num_classes, args.hidden_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best = {"ap": -1.0, "epoch": 0, "hit1": 0.0, "hit2": 0.0, "state": None}
    n = x_train.shape[0]
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total = 0.0
        for i in range(0, n, args.batch_size):
            idx = perm[i:i + args.batch_size].to(device)
            optimizer.zero_grad()
            loss = criterion(model(x_train[idx]), y_train[idx])
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)

        ap, hit1, hit2 = evaluate(model, x_val, y_val, device)
        if ap > best["ap"]:
            best = {"ap": ap, "epoch": epoch, "hit1": hit1, "hit2": hit2,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
        if epoch % 5 == 0 or epoch == 1:
            print(f"[{args.mode} seed{args.seed}] epoch {epoch}/{args.epochs} "
                  f"train_loss={total / n:.4f} val_macro_ap={ap:.4f}")

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    ckpt = CHECKPOINT_DIR / f"movement_cached_{args.mode}_seed{args.seed}.pt"
    torch.save(best["state"], ckpt)

    print(f"RESULT mode={args.mode} seed={args.seed} params={n_params:,} "
          f"best_epoch={best['epoch']} val_macro_ap={best['ap']:.4f} "
          f"val_top1={best['hit1']:.3f} val_top2={best['hit2']:.3f}")

    if args.eval_test:
        model.load_state_dict(best["state"])
        x_test = build_features(cache["test"]["embeds"], args.mode)
        ap, hit1, hit2 = evaluate(model, x_test, cache["test"]["labels"], device)
        print(f"TEST   mode={args.mode} seed={args.seed} "
              f"test_macro_ap={ap:.4f} test_top1={hit1:.3f} test_top2={hit2:.3f}")


if __name__ == "__main__":
    main()