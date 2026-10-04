"""
src/training/cache_static_embeddings.py

Runs the FROZEN CLIP image encoder once over every image in the train / val /
test splits and saves the embeddings (plus labels) to disk. Heads can then be
trained and compared in minutes instead of re-running CLIP every epoch.

This is equivalent to what StaticShotClassifier.forward() computes before its
heads (vision_model -> pooler_output -> visual_projection), done in fp32.

Images that cannot be read are skipped and counted (the per-split count is
printed and saved), so the script does not crash on a bad file.

Output: data/processed/static_embeddings_<clip name>.pt (data/processed/ is
gitignored, so it is not committed). With --max-samples the output name ends
in _smoke.pt so a test run never overwrites the real cache.

Usage:
    python -m src.training.cache_static_embeddings --max-samples 256
    python -m src.training.cache_static_embeddings
    python -m src.training.cache_static_embeddings --clip-checkpoint openai/clip-vit-large-patch14
"""

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from src.data.dataset import ShotDataset
from src.data.label_encoding import CLASS_VOCAB
from src.models.static_classifier import DEFAULT_CLIP_CHECKPOINT, StaticShotClassifier

DIMENSIONS = list(CLASS_VOCAB.keys())
SPLITS = ("train", "val", "test")
OUT_DIR = Path("data/processed")
IMAGE_SHAPE = (3, 224, 224)  # all CLIP checkpoints used here take 224x224 input


class SafeDataset(Dataset):
    """Wraps ShotDataset. A bad image returns zeros plus ok=False instead of crashing."""

    def __init__(self, base, img_ids):
        self.base = base
        self.img_ids = img_ids

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        try:
            image, labels = self.base[idx]
            return image, labels, True
        except Exception:
            labels = {d: torch.zeros(len(CLASS_VOCAB[d])) for d in DIMENSIONS}
            return torch.zeros(IMAGE_SHAPE), labels, False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-checkpoint", type=str, default=DEFAULT_CLIP_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Only the first N images per split (smoke test).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    tag = args.clip_checkpoint.split("/")[-1]
    suffix = "_smoke" if args.max_samples else ""
    out_path = OUT_DIR / f"static_embeddings_{tag}{suffix}.pt"

    model = StaticShotClassifier(CLASS_VOCAB, clip_checkpoint=args.clip_checkpoint).to(device)
    model.eval()
    transform = model.get_preprocess()

    cache = {
        "clip_checkpoint": args.clip_checkpoint,
        "dimensions": DIMENSIONS,
        "class_vocab": {d: list(CLASS_VOCAB[d]) for d in DIMENSIONS},
    }

    for split in SPLITS:
        ds = ShotDataset(split=split, transform=transform)
        img_ids = list(ds.img_ids)
        if args.max_samples:
            n = min(args.max_samples, len(ds))
            base = Subset(ds, range(n))
            img_ids = img_ids[:n]
        else:
            base = ds

        loader = DataLoader(
            SafeDataset(base, img_ids), batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
        )
        print(f"\n{split}: {len(img_ids)} images")

        embeds, ok_flags = [], []
        labels = {d: [] for d in DIMENSIONS}
        start, done = time.time(), 0
        with torch.no_grad():
            for i, (images, lab, ok) in enumerate(loader, start=1):
                vision = model.clip.vision_model(pixel_values=images.to(device))
                feats = model.clip.visual_projection(vision.pooler_output)
                embeds.append(feats.float().cpu())
                for d in DIMENSIONS:
                    labels[d].append(lab[d])
                ok_flags.append(ok)
                done += images.shape[0]
                if i % 100 == 0 or i == len(loader):
                    rate = done / (time.time() - start)
                    eta = (len(img_ids) - done) / rate
                    print(f"  batch {i}/{len(loader)}  {rate:.1f} img/s  ETA {eta:.0f}s")

        mask = torch.cat(ok_flags)
        kept_ids = [img for img, m in zip(img_ids, mask.tolist()) if m]
        n_bad = len(img_ids) - len(kept_ids)
        embeds = torch.cat(embeds)[mask]
        cache[split] = {
            "img_ids": kept_ids,
            "embeds": embeds,
            "labels": {d: torch.cat(labels[d])[mask] for d in DIMENSIONS},
            "n_unreadable": n_bad,
        }
        print(f"  kept {len(kept_ids)}, unreadable {n_bad}, embeds {tuple(embeds.shape)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(cache, out_path)
    print(f"\nSaved {out_path} ({out_path.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()