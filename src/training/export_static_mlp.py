"""
src/training/export_static_mlp.py

Turns a trained experiment file (made by train_static_cached.py) into the
deployable pair used by inference:
    checkpoints/best_model_mlp.pt         (heads only + settings, small file)
    reports/best_thresholds_mlp.json      (per-class thresholds tuned on val)

Before saving anything it checks that the REAL pipeline (image file ->
StaticShotClassifier with the loaded heads) gives the same logits as the
cached embeddings did, on the first N val images. If they differ by more
than --tol, nothing is saved.

It never writes best_model.pt or best_thresholds.json (the old model stays
as a fallback) and refuses to overwrite its own outputs unless --force.

Usage:
    python -m src.training.export_static_mlp
    python -m src.training.export_static_mlp --experiment checkpoints/experiments/<file>.pt
"""

import argparse
import json
from pathlib import Path

import torch

from src.data.dataset import ShotDataset
from src.data.label_encoding import CLASS_VOCAB
from src.models.static_classifier import StaticShotClassifier
from src.training.static_cached_common import DEFAULT_CACHE, load_cache

DEFAULT_EXPERIMENT = Path(
    "checkpoints/experiments/"
    "static_clip-vit-base-patch32_mlp_seed1_ep10_lr0.0003_wd0_do0.5_h512.pt"
)
PROTECTED_NAMES = {"best_model.pt", "best_thresholds.json"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=Path, default=DEFAULT_EXPERIMENT)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--out-checkpoint", type=Path, default=Path("checkpoints/best_model_mlp.pt"))
    parser.add_argument("--out-thresholds", type=Path, default=Path("reports/best_thresholds_mlp.json"))
    parser.add_argument("--n-verify", type=int, default=64)
    parser.add_argument("--tol", type=float, default=1e-3)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for p in (args.out_checkpoint, args.out_thresholds):
        if p.name in PROTECTED_NAMES:
            raise SystemExit(f"Refusing to write {p}: that is the old model's file.")
        if p.exists() and not args.force:
            raise SystemExit(f"{p} already exists. Use --force to overwrite it.")

    if not args.experiment.exists():
        raise SystemExit(f"{args.experiment} not found.")
    # Our own file (written by train_static_cached.py), so full unpickling is fine.
    exp = torch.load(args.experiment, map_location="cpu", weights_only=False)

    run_args = exp["args"]
    head_type = run_args["head"]
    hidden = int(run_args["hidden"])
    dropout = float(run_args["dropout"])
    clip_ckpt = exp.get("clip_checkpoint")
    if not clip_ckpt:
        raise SystemExit("Experiment file has no clip_checkpoint (made by an older script).")

    heads_state = {k[len("heads."):]: v for k, v in exp["model_state_dict"].items()
                   if k.startswith("heads.")}
    thresholds = exp["thresholds"]
    for dim, classes in CLASS_VOCAB.items():
        if len(thresholds[dim]) != len(classes):
            raise SystemExit(f"Threshold count mismatch for {dim}.")

    print(f"Experiment: {args.experiment.name}")
    print(f"Backbone: {clip_ckpt} | head: {head_type} (hidden {hidden}, dropout {dropout}) | "
          f"best epoch {exp['best_epoch']}")
    val_ap = exp["val_macro_ap"]
    print(f"Val mean macro-AP stored in the file: {sum(val_ap.values()) / len(val_ap):.4f}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = StaticShotClassifier(CLASS_VOCAB, clip_checkpoint=clip_ckpt,
                                 head_type=head_type, hidden=hidden, dropout=dropout)
    model.heads.load_state_dict(heads_state)  # strict: errors on any mismatch
    model.to(device)
    model.eval()
    print("Heads loaded into StaticShotClassifier (strict).")

    # --- Check: real image path vs cached embeddings ---
    cache = load_cache(args.cache)
    if cache.get("clip_checkpoint") != clip_ckpt:
        raise SystemExit(f"Cache backbone {cache.get('clip_checkpoint')} != {clip_ckpt}.")
    ds = ShotDataset(split="val", transform=model.get_preprocess())
    n = min(args.n_verify, len(ds), cache["val"]["embeds"].shape[0])
    ids_match = list(ds.img_ids[:n]) == list(cache["val"]["img_ids"][:n])
    print(f"\nChecking {n} val images. Image ids line up with the cache: {ids_match}")
    if not ids_match:
        raise SystemExit("Dataset order differs from the cache; cannot compare. Nothing saved.")

    max_diff = 0.0
    with torch.no_grad():
        for i in range(n):
            image, _ = ds[i]
            out = model(image.unsqueeze(0).to(device))
            emb = cache["val"]["embeds"][i:i + 1].to(device)
            for d in CLASS_VOCAB:
                ref = model.heads[d](emb)
                max_diff = max(max_diff, (out[d] - ref).abs().max().item())
    print(f"Largest logit difference (real pipeline vs cached embeddings): {max_diff:.2e} "
          f"(tolerance {args.tol:.0e})")
    if max_diff > args.tol:
        raise SystemExit("MISMATCH: nothing saved. Paste this output so we can find the cause.")
    print("OK: the real pipeline reproduces the cached results.")

    # --- Save ---
    args.out_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    args.out_thresholds.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "head_type": head_type,
        "hidden": hidden,
        "dropout": dropout,
        "clip_checkpoint": clip_ckpt,
        "dimensions": list(CLASS_VOCAB.keys()),
        "heads_state_dict": {k: v.cpu() for k, v in heads_state.items()},
        "best_epoch": exp["best_epoch"],
        "source_experiment": args.experiment.name,
        "val_macro_ap": exp["val_macro_ap"],
        "val_macro_f1": exp["val_macro_f1"],
    }, args.out_checkpoint)
    with open(args.out_thresholds, "w") as f:
        json.dump(thresholds, f, indent=2)
    print(f"\nSaved {args.out_checkpoint} ({args.out_checkpoint.stat().st_size / 1e6:.1f} MB)")
    print(f"Saved {args.out_thresholds}")


if __name__ == "__main__":
    main()