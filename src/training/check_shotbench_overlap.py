"""
src/training/check_shotbench_overlap.py

Checks whether ShotBench IMAGES have near-duplicate (or same-scene) twins
in the ShotQA images the model was trained on.

How:
  1. Turn every ShotQA image and every ShotBench image into a CLIP
     fingerprint (a list of 512 numbers), using the same frozen CLIP
     that is inside the static model.
  2. For each ShotBench image, find the most similar ShotQA image
     (cosine similarity: 1.0 = identical, lower = less alike).
  3. Group ShotBench image questions by that similarity and show the
     model's accuracy per group.

If accuracy is much higher for images that have a very close twin, the
overall score is partly memorisation. If it is about the same, that is
evidence (not proof) that overlap is not driving the result.

Does NOT cover the 464 movement videos.

Fingerprints are cached on disk. If new images appear in a folder, only
the new ones are embedded (the old cache is reused).

Usage:
    python -m src.training.check_shotbench_overlap
"""

import argparse
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from src.inference.video_analyzer import load_static_model
from src.models.static_classifier import ClipPreprocess, DEFAULT_CLIP_CHECKPOINT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHOTQA_DIR = PROJECT_ROOT / "data" / "raw" / "images"
SHOTBENCH_DIR = PROJECT_ROOT / "data" / "raw" / "shotbench"
CACHE_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

QA_CACHE = CACHE_DIR / "overlap_shotqa_embeddings.npz"
SB_CACHE = CACHE_DIR / "overlap_shotbench_embeddings.npz"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


class ImageListDataset(Dataset):
    def __init__(self, paths, preprocess):
        self.paths = paths
        self.preprocess = preprocess

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        try:
            img = Image.open(self.paths[i]).convert("RGB")
            return self.preprocess(img), True
        except Exception:
            return torch.zeros(3, 224, 224), False


def embed_images(paths, model, preprocess, device, num_workers, desc):
    ds = ImageListDataset(paths, preprocess)
    dl = DataLoader(ds, batch_size=64, shuffle=False, num_workers=num_workers)
    feats, oks = [], []
    for pixel_values, ok in tqdm(dl, desc=desc):
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            vision_out = model.clip.vision_model(pixel_values=pixel_values)
            f = model.clip.visual_projection(vision_out.pooler_output)
            f = torch.nn.functional.normalize(f, dim=-1)
        feats.append(f.cpu().numpy().astype(np.float32))
        oks.append(ok.numpy())
    return np.concatenate(feats), np.concatenate(oks)


def get_embeddings(cache_path, names, dir_root, model, preprocess, device, num_workers, desc):
    """Returns (feats, ok) in the same order as `names`. Only embeds names not already cached."""
    cached = {}
    if cache_path.exists():
        data = np.load(cache_path)
        for n, f, o in zip(data["names"], data["feats"], data["ok"]):
            cached[str(n)] = (f, bool(o))

    missing = [n for n in names if n not in cached]
    if missing:
        print(f"{desc}: embedding {len(missing)} image(s) not in the cache...")
        paths = [str(dir_root / n) for n in missing]
        feats, ok = embed_images(paths, model, preprocess, device, num_workers, desc)
        for n, f, o in zip(missing, feats, ok):
            cached[n] = (f, bool(o))
        all_names = sorted(cached)
        np.savez(
            cache_path,
            feats=np.stack([cached[n][0] for n in all_names]),
            ok=np.array([cached[n][1] for n in all_names]),
            names=np.array(all_names),
        )
    else:
        print(f"{desc}: all {len(names)} loaded from cache.")

    feats = np.stack([cached[n][0] for n in names])
    ok = np.array([cached[n][1] for n in names])
    return feats, ok


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ---- which ShotBench files are images, and which question uses which file
    df = pd.read_csv(SHOTBENCH_DIR / "test.tsv", sep="\t")
    df["type_list"] = df["type"].apply(lambda x: ast.literal_eval(str(x)))
    df["path_list"] = df["path"].apply(lambda x: ast.literal_eval(str(x)))
    df = df[df["type_list"].apply(lambda t: len(t) == 1 and t[0] == "image")].copy()
    df["file"] = df["path_list"].apply(lambda x: x[0])
    sb_names = sorted(df["file"].unique())
    print(f"ShotBench image questions: {len(df)}, unique image files: {len(sb_names)}")

    qa_names = sorted(
        p.name for p in SHOTQA_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    print(f"ShotQA images found (jpg + jpeg + png): {len(qa_names)}")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading model...")
    model = load_static_model(device)
    preprocess = ClipPreprocess(DEFAULT_CLIP_CHECKPOINT)

    sb_feats, sb_ok = get_embeddings(
        SB_CACHE, sb_names, SHOTBENCH_DIR, model, preprocess, device,
        args.num_workers, "ShotBench images")
    qa_feats, qa_ok = get_embeddings(
        QA_CACHE, qa_names, SHOTQA_DIR, model, preprocess, device,
        args.num_workers, "ShotQA images")

    print(f"ShotBench images unreadable: {int((~sb_ok).sum())}")
    print(f"ShotQA images unreadable: {int((~qa_ok).sum())}")
    unreadable = [qa_names[i] for i in np.where(~qa_ok)[0]]
    pd.DataFrame({"unreadable_shotqa_file": unreadable}).to_csv(
        REPORTS_DIR / "overlap_unreadable_shotqa.csv", index=False)

    # ---- nearest ShotQA image for every ShotBench image
    qa_keep = np.where(qa_ok)[0]
    qa_t = torch.tensor(qa_feats[qa_keep], device=device)
    sb_t = torch.tensor(sb_feats, device=device)

    best_sim = np.zeros(len(sb_names), dtype=np.float32)
    best_idx = np.zeros(len(sb_names), dtype=int)
    for start in range(0, len(sb_names), 256):
        chunk = sb_t[start:start + 256]
        sims = chunk @ qa_t.T
        vals, idxs = sims.max(dim=1)
        best_sim[start:start + 256] = vals.cpu().numpy()
        best_idx[start:start + 256] = qa_keep[idxs.cpu().numpy()]

    nn_df = pd.DataFrame({
        "shotbench_file": sb_names,
        "nearest_shotqa_file": [qa_names[i] for i in best_idx],
        "max_similarity": best_sim,
        "shotbench_readable": sb_ok,
    })
    nn_df = nn_df.sort_values("max_similarity", ascending=False)
    nn_df.to_csv(REPORTS_DIR / "shotbench_overlap_nearest.csv", index=False)

    print("\n=== How similar is each ShotBench image to its closest ShotQA image? ===")
    s = nn_df.loc[nn_df["shotbench_readable"], "max_similarity"]
    for q in [0.10, 0.25, 0.50, 0.75, 0.90, 0.99]:
        print(f"  {int(q * 100):3d}th percentile: {s.quantile(q):.4f}")
    print(f"  highest: {s.max():.4f}")
    for thr in [0.90, 0.95, 0.98]:
        print(f"  images with similarity >= {thr}: {int((s >= thr).sum())} of {len(s)}")

    print("\n=== 8 most similar pairs ===")
    for _, r in nn_df.head(8).iterrows():
        print(f"  sim {r['max_similarity']:.4f}")
        print(f"     ShotBench: {SHOTBENCH_DIR / r['shotbench_file']}")
        print(f"     ShotQA:    {SHOTQA_DIR / r['nearest_shotqa_file']}")

    # ---- accuracy by similarity group
    preds = pd.read_csv(REPORTS_DIR / "shotbench_predictions.csv")
    preds["index"] = preds["index"].astype(str)
    df["index"] = df["index"].astype(str)
    merged = preds.merge(df[["index", "file"]], on="index", how="inner")
    merged = merged.merge(
        nn_df[["shotbench_file", "max_similarity"]],
        left_on="file", right_on="shotbench_file", how="left")
    merged = merged.dropna(subset=["max_similarity"])

    bins = [0.0, 0.85, 0.90, 0.95, 1.01]
    labels = ["below 0.85", "0.85-0.90", "0.90-0.95", "0.95 and above"]
    merged["group"] = pd.cut(merged["max_similarity"], bins=bins, labels=labels, right=False)

    print("\n=== Model accuracy by similarity to ShotQA (image questions only) ===")
    overall = merged.groupby("group", observed=False)["correct_credit"].agg(["size", "mean"])
    overall.columns = ["n_questions", "accuracy"]
    overall["accuracy"] = overall["accuracy"].round(4)
    print(overall.to_string())
    print(f"\nAll image questions: n={len(merged)}, accuracy={merged['correct_credit'].mean():.4f}")

    by_cat = merged.pivot_table(index="category", columns="group",
                                values="correct_credit", aggfunc=["size", "mean"], observed=False)
    by_cat.to_csv(REPORTS_DIR / "shotbench_overlap_by_category.csv")
    merged.to_csv(REPORTS_DIR / "shotbench_overlap_merged.csv", index=False)

    print("\nSaved to reports/: shotbench_overlap_nearest.csv, shotbench_overlap_by_category.csv, "
          "shotbench_overlap_merged.csv, overlap_unreadable_shotqa.csv")
    print("Similarity cut-offs (0.85 / 0.90 / 0.95) are arbitrary starting points, not proven thresholds.")


if __name__ == "__main__":
    main()