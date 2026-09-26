"""
Video Analyzer (Module 1 inference wrapper). Given a video:
  1. Segments it into shots (PySceneDetect via shot_segmentation.py)
  2. For each shot, runs the static 7-dimension classifier on a
     representative (middle) frame
  3. For each shot, runs the movement LSTM classifier on 16 uniformly
     sampled frames from within that shot
Produces a per-shot dict of predicted labels — the raw material Module 2
(Script Generator) will format into a shot-by-shot breakdown.
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from src.inference.shot_segmentation import segment_video
from src.models.static_classifier import (
    StaticShotClassifier,
    ClipPreprocess,
    DEFAULT_CLIP_CHECKPOINT,
)
from src.models.movement_classifier import MovementClassifier
from src.data.label_encoding import CLASS_VOCAB
from src.data.movement_label_encoding import MOVEMENT_CLASS_VOCAB

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STATIC_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_model.pt"
STATIC_THRESHOLDS = PROJECT_ROOT / "reports" / "best_thresholds.json"
MOVEMENT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "movement_lstm.pt"
MOVEMENT_THRESHOLDS = PROJECT_ROOT / "reports" / "movement_best_thresholds.json"

N_MOVEMENT_FRAMES = 16
MOVEMENT_CLASS_NAMES = list(MOVEMENT_CLASS_VOCAB)


def load_static_model(device):
    model = StaticShotClassifier(class_vocab=CLASS_VOCAB)
    ckpt = torch.load(STATIC_CHECKPOINT, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_movement_model(device):
    model = MovementClassifier(num_classes=len(MOVEMENT_CLASS_NAMES), temporal_type="lstm")
    sd = torch.load(MOVEMENT_CHECKPOINT, map_location=device)
    if isinstance(sd, dict) and "model_state_dict" in sd:
        sd = sd["model_state_dict"]
    model.load_state_dict(sd)
    model.to(device)
    model.eval()
    return model


def decode_all_frames(video_path):
    """
    Decode every frame of a video into memory as PIL Images. Avoids
    OpenCV's seek-based frame access (cap.set(CAP_PROP_POS_FRAMES, ...)),
    which is unreliable on some codecs -- the same reason
    extract_movement_frames.py decodes fully in memory instead of seeking.
    """
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
    cap.release()
    return frames


def classify_static(model, preprocess, frame_img, thresholds, device):
    pixel_values = preprocess(frame_img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(pixel_values)

    predictions = {}
    for dim, logits in outputs.items():
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        dim_thresholds = thresholds[dim]
        class_names = CLASS_VOCAB[dim]
        predicted = [class_names[i] for i, p in enumerate(probs) if p >= dim_thresholds[i]]
        # fall back to the single highest-probability class if nothing crosses threshold
        predictions[dim] = predicted if predicted else [class_names[int(np.argmax(probs))]]
    return predictions


def classify_movement(model, preprocess, frame_imgs, thresholds_dict, device):
    pixel_values = torch.stack([preprocess(f) for f in frame_imgs])  # (T, C, H, W)
    pixel_values = pixel_values.unsqueeze(0).to(device)  # (1, T, C, H, W)
    with torch.no_grad():
        logits = model(pixel_values)
    probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    predicted = [
        name for name, p in zip(MOVEMENT_CLASS_NAMES, probs)
        if p >= thresholds_dict.get(name, 0.5)
    ]
    return predicted if predicted else ["Static shot"]


def analyze_video(video_path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading models...")
    static_model = load_static_model(device)
    movement_model = load_movement_model(device)
    preprocess = ClipPreprocess(DEFAULT_CLIP_CHECKPOINT)

    with open(STATIC_THRESHOLDS) as f:
        static_thresholds = json.load(f)
    with open(MOVEMENT_THRESHOLDS) as f:
        movement_thresholds = json.load(f)

    print(f"Segmenting: {video_path}")
    shots = segment_video(str(video_path))
    print(f"Found {len(shots)} shot(s)")

    print("Decoding all frames...")
    all_frames = decode_all_frames(video_path)
    print(f"Decoded {len(all_frames)} frames")

    results = []
    for shot in shots:
        start_f = shot["start_frame"]
        end_f = min(shot["end_frame"], len(all_frames))
        if end_f <= start_f:
            print(f"  Skipping shot {shot['shot_index']} — invalid frame range")
            continue

        mid_idx = (start_f + end_f) // 2
        rep_frame = all_frames[mid_idx]
        static_preds = classify_static(static_model, preprocess, rep_frame, static_thresholds, device)

        indices = np.linspace(start_f, end_f - 1, num=N_MOVEMENT_FRAMES, dtype=int)
        shot_frames = [all_frames[i] for i in indices]
        movement_preds = classify_movement(movement_model, preprocess, shot_frames, movement_thresholds, device)

        results.append({
            "shot_index": shot["shot_index"],
            "start_timecode": shot["start_timecode"],
            "end_timecode": shot["end_timecode"],
            "static": static_preds,
            "movement": movement_preds,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description="Analyze a video: segment into shots, classify each.")
    parser.add_argument("video_path", type=str)
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    results = analyze_video(video_path)

    print("\n=== Shot-by-shot analysis ===\n")
    for r in results:
        print(f"Shot {r['shot_index']}: {r['start_timecode']} -> {r['end_timecode']}")
        for dim, vals in r["static"].items():
            print(f"  {dim}: {', '.join(vals)}")
        print(f"  Movement: {', '.join(r['movement'])}")
        print()


if __name__ == "__main__":
    main()