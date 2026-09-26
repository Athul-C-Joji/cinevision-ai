"""
src/data/extract_movement_frames.py

Extracts 16 uniformly-spaced frames from each camera-movement video clip,
for use as the input sequence to the temporal movement model.

Reads clip filenames from data/processed/movement_labels_encoded.csv,
locates the source video files in data/raw/images/, and saves extracted
frames as JPEGs under data/processed/movement_frames/<clip_id>/frame_XX.jpg.

Frames are saved at native resolution -- no CLIP preprocessing is applied
here. Resizing/cropping/normalization happens later in the Dataset class,
mirroring the ClipPreprocess pattern already used in static_classifier.py.

Each clip is fully decoded into memory before sampling (rather than seeking
by frame index), since OpenCV's seek-by-index can drift on some codecs.
These clips are short, so this is cheap.

Usage:
    python -m src.data.extract_movement_frames
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

N_FRAMES = 16

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LABELS_CSV = PROJECT_ROOT / "data" / "processed" / "movement_labels_encoded.csv"
VIDEOS_DIR = PROJECT_ROOT / "data" / "raw" / "images"
OUT_DIR = PROJECT_ROOT / "data" / "processed" / "movement_frames"
REPORT_PATH = PROJECT_ROOT / "reports" / "movement_frame_extraction.csv"


def sample_frame_indices(total_frames: int, n: int) -> list[int]:
    """Evenly spaced frame indices across the clip. If the clip has fewer
    frames than n, indices repeat so we still return n frames."""
    if total_frames <= 0:
        return [0] * n
    if total_frames >= n:
        return list(np.linspace(0, total_frames - 1, n).round().astype(int))
    return [min(i * total_frames // n, total_frames - 1) for i in range(n)]


def extract_frames_for_clip(video_path: Path, out_dir: Path, n: int) -> tuple[int, int]:
    """Returns (frames_saved, total_frames_decoded)."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0, 0

    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()

    total_frames = len(frames)
    if total_frames == 0:
        return 0, 0

    indices = sample_frame_indices(total_frames, n)
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = 0
    for slot, idx in enumerate(indices):
        out_path = out_dir / f"frame_{slot:02d}.jpg"
        cv2.imwrite(str(out_path), frames[idx])
        saved += 1

    return saved, total_frames


def main():
    df = pd.read_csv(LABELS_CSV)
    filenames = df["filename"].tolist()

    report_rows = []
    n_ok = 0
    n_missing_video = 0
    n_short_clips = 0
    n_decode_failed = 0

    for fname in filenames:
        clip_id = Path(fname).stem
        video_path = VIDEOS_DIR / fname
        clip_out_dir = OUT_DIR / clip_id

        if not video_path.exists():
            n_missing_video += 1
            report_rows.append({
                "filename": fname,
                "status": "MISSING_VIDEO",
                "frames_saved": 0,
                "total_frames_decoded": 0,
            })
            continue

        saved, total = extract_frames_for_clip(video_path, clip_out_dir, N_FRAMES)

        if total == 0:
            n_decode_failed += 1
            status = "DECODE_FAILED"
        elif total < N_FRAMES:
            n_short_clips += 1
            status = f"SHORT_CLIP ({total} frames, repeated to fill {N_FRAMES})"
            n_ok += 1
        elif saved == N_FRAMES:
            status = "OK"
            n_ok += 1
        else:
            status = f"PARTIAL ({saved}/{N_FRAMES} saved)"

        report_rows.append({
            "filename": fname,
            "status": status,
            "frames_saved": saved,
            "total_frames_decoded": total,
        })

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(report_rows).to_csv(REPORT_PATH, index=False)

    print(f"Total clips: {len(filenames)}")
    print(f"Successfully extracted {N_FRAMES} frames: {n_ok}")
    print(f"Missing video files: {n_missing_video}")
    print(f"Decode failures (0 frames readable): {n_decode_failed}")
    print(f"Short clips (fewer than {N_FRAMES} frames, repeated to fill): {n_short_clips}")
    print(f"Frames saved under: {OUT_DIR}")
    print(f"Full report: {REPORT_PATH}")


if __name__ == "__main__":
    main()