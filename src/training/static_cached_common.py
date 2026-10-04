"""
src/training/static_cached_common.py

Shared pieces for training and evaluating the 7 static heads on CACHED CLIP
embeddings (made by cache_static_embeddings.py):
  - StaticHeads: same layout as the deployed heads when head_type="linear"
    (state-dict keys are "heads.<dimension>.weight" / ".bias", the same as in
    checkpoints/best_model.pt), or a small hidden layer when head_type="mlp".
  - metrics: macro average precision (threshold-free), per-class threshold
    tuning copied from find_best_thresholds.py (grid 0.05..0.90, max F1 per
    class, first best kept), and macro-F1 at given thresholds.
"""

from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

DEFAULT_CACHE = Path("data/processed/static_embeddings_clip-vit-base-patch32.pt")
THRESHOLD_GRID = np.arange(0.05, 0.95, 0.05)


def load_cache(path):
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"{path} not found. Run: python -m src.training.cache_static_embeddings")
    return torch.load(path, weights_only=True)


class StaticHeads(nn.Module):
    def __init__(self, in_dim, class_vocab, head_type="linear", hidden=512, dropout=0.2):
        super().__init__()

        def make(n_classes):
            if head_type == "linear":
                return nn.Linear(in_dim, n_classes)
            return nn.Sequential(
                nn.Linear(in_dim, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, n_classes),
            )

        self.heads = nn.ModuleDict(OrderedDict(
            (dim, make(len(classes))) for dim, classes in class_vocab.items()
        ))

    def forward(self, x):
        return {dim: head(x) for dim, head in self.heads.items()}


def predict_logits(model, x, device, batch=2048):
    model.eval()
    outs = {}
    with torch.no_grad():
        for i in range(0, x.shape[0], batch):
            out = model(x[i:i + batch].to(device))
            for d, logits in out.items():
                outs.setdefault(d, []).append(logits.cpu())
    return {d: torch.cat(v) for d, v in outs.items()}


def collect(model, split, device, dims):
    """Returns (probs, targets, loss) for one split. loss = sum over heads of the
    mean BCE-with-logits, the same quantity train.py logs."""
    logits = predict_logits(model, split["embeds"], device)
    loss = sum(F.binary_cross_entropy_with_logits(logits[d], split["labels"][d]).item()
               for d in dims)
    probs = {d: torch.sigmoid(logits[d]).numpy() for d in dims}
    targets = {d: split["labels"][d].numpy().astype(int) for d in dims}
    return probs, targets, loss


def macro_ap(probs, targets):
    aps = [average_precision_score(targets[:, c], probs[:, c])
           for c in range(targets.shape[1]) if targets[:, c].sum() > 0]
    return float(np.mean(aps)) if aps else 0.0


def f1_at(p, t, thr):
    pred = p >= thr
    tp = int(np.sum(pred & (t == 1)))
    fp = int(np.sum(pred & (t == 0)))
    fn = int(np.sum(~pred & (t == 1)))
    denom = 2 * tp + fp + fn
    return 0.0 if denom == 0 else 2 * tp / denom


def tune_thresholds(probs, targets, dims):
    thresholds = {}
    for d in dims:
        per_class = []
        for c in range(targets[d].shape[1]):
            best_f1, best_t = -1.0, 0.5
            for thr in THRESHOLD_GRID:
                f1 = f1_at(probs[d][:, c], targets[d][:, c], thr)
                if f1 > best_f1:
                    best_f1, best_t = f1, float(thr)
            per_class.append(best_t)
        thresholds[d] = per_class
    return thresholds


def macro_f1(probs, targets, thresholds):
    f1s = [f1_at(probs[:, c], targets[:, c], thresholds[c]) for c in range(targets.shape[1])]
    return float(np.mean(f1s))


def score_split(probs, targets, thresholds, dims):
    ap = {d: macro_ap(probs[d], targets[d]) for d in dims}
    f1 = {d: macro_f1(probs[d], targets[d], thresholds[d]) for d in dims}
    return ap, f1


def print_scores(title, ap, f1, dims, targets=None):
    print(f"\n{title}")
    print(f"  {'Dimension':<16}{'macro_AP':>10}{'macro_F1':>10}")
    for d in dims:
        print(f"  {d:<16}{ap[d]:>10.3f}{f1[d]:>10.3f}")
    print(f"  {'MEAN':<16}{np.mean(list(ap.values())):>10.3f}{np.mean(list(f1.values())):>10.3f}")
    if targets is not None:
        zero = sum(int((targets[d].sum(axis=0) == 0).sum()) for d in dims)
        if zero:
            print(f"  note: {zero} class(es) have no positive examples in this split "
                  f"(F1 counted as 0, AP skipped)")