"""
Per-class threshold tuning for the movement LSTM classifier.
Mirrors src/training/find_best_thresholds.py from Module 1, but operates on
clip-level sequence inputs (16 frames/clip) and the 21-class movement vocab.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import f1_score, precision_score, recall_score

from src.data.movement_dataset import MovementDataset
from src.data.movement_label_encoding import MOVEMENT_CLASS_VOCAB
from src.models.movement_classifier import MovementClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "movement_lstm.pt"
SPLITS_CSV = PROJECT_ROOT / "data" / "processed" / "movement_splits.csv"
LABELS_CSV = PROJECT_ROOT / "data" / "processed" / "movement_labels_encoded.csv"
FRAMES_DIR = PROJECT_ROOT / "data" / "processed" / "movement_frames"
OUTPUT_THRESHOLDS = PROJECT_ROOT / "reports" / "movement_best_thresholds.json"
OUTPUT_METRICS = PROJECT_ROOT / "reports" / "movement_eval_metrics_tuned.csv"

THRESHOLD_GRID = np.arange(0.05, 1.00, 0.05)
CLASS_NAMES = list(MOVEMENT_CLASS_VOCAB)
NUM_CLASSES = len(CLASS_NAMES)

# Standard CLIP (openai/clip-vit-base-patch32) preprocessing, defined here
# directly so this script doesn't depend on guessing an internal class name
# from static_classifier.py. Applied per-frame.
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

clip_transform = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
])


def get_val_predictions(model, val_loader, device):
    """Run the model on the val split once, return (all_probs, all_labels) as numpy arrays."""
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for frames, labels in val_loader:
            frames = frames.to(device)
            logits = model(frames)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_labels, axis=0)


def tune_thresholds(all_probs, all_labels):
    """For each class, sweep THRESHOLD_GRID and keep whichever maximizes that class's F1."""
    best_thresholds = {}
    rows = []

    for class_idx, class_name in enumerate(CLASS_NAMES):
        y_true = all_labels[:, class_idx]
        y_prob = all_probs[:, class_idx]
        support = int(y_true.sum())

        best_f1 = -1.0
        best_t = 0.5
        for t in THRESHOLD_GRID:
            y_pred = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_t = round(float(t), 2)

        y_pred_best = (y_prob >= best_t).astype(int)
        precision = precision_score(y_true, y_pred_best, zero_division=0)
        recall = recall_score(y_true, y_pred_best, zero_division=0)

        y_pred_flat = (y_prob >= 0.5).astype(int)
        flat_f1 = f1_score(y_true, y_pred_flat, zero_division=0)

        best_thresholds[class_name] = best_t
        rows.append({
            "class": class_name,
            "support": support,
            "flat_0.5_f1": round(flat_f1, 4),
            "tuned_threshold": best_t,
            "tuned_precision": round(precision, 4),
            "tuned_recall": round(recall, 4),
            "tuned_f1": round(best_f1, 4),
        })

    return best_thresholds, pd.DataFrame(rows)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    val_dataset = MovementDataset(
        split="val",
        transform=clip_transform,
        splits_csv=SPLITS_CSV,
        labels_csv=LABELS_CSV,
        frames_dir=FRAMES_DIR,
        n_frames=16,
    )
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False, num_workers=4)
    print(f"Val clips: {len(val_dataset)}")

    model = MovementClassifier(num_classes=NUM_CLASSES, temporal_type="lstm")
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)

    print("Running inference on val split...")
    all_probs, all_labels = get_val_predictions(model, val_loader, device)

    print("Sweeping per-class thresholds...")
    best_thresholds, metrics_df = tune_thresholds(all_probs, all_labels)

    OUTPUT_THRESHOLDS.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_THRESHOLDS, "w") as f:
        json.dump(best_thresholds, f, indent=2)
    metrics_df.to_csv(OUTPUT_METRICS, index=False)

    macro_f1_flat = metrics_df["flat_0.5_f1"].mean()
    macro_f1_tuned = metrics_df["tuned_f1"].mean()

    supported = metrics_df[metrics_df["support"] > 0]
    macro_f1_tuned_supported = supported["tuned_f1"].mean()
    n_zero_support = (metrics_df["support"] == 0).sum()

    print(f"\nSaved thresholds -> {OUTPUT_THRESHOLDS}")
    print(f"Saved per-class metrics -> {OUTPUT_METRICS}")
    print(f"\nMacro-F1 flat 0.5:                  {macro_f1_flat:.4f}")
    print(f"Macro-F1 tuned (all {NUM_CLASSES} classes):      {macro_f1_tuned:.4f}")
    print(f"Macro-F1 tuned ({len(supported)} classes w/ val support): {macro_f1_tuned_supported:.4f}")
    print(f"Classes with zero val support: {n_zero_support} -> {list(metrics_df[metrics_df['support']==0]['class'])}")
    print("\nPer-class results:")
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()