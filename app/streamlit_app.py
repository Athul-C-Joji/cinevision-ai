"""
app/streamlit_app.py

CineVision AI demo: upload a video, get a shot-by-shot breakdown.

Uses the same building blocks as src/inference/video_analyzer.py
(analyze_video), in the same order, so the app can also keep a thumbnail
per shot and load the models only once. If analyze_video changes, this
file needs the same change.

Run from the project root:
    python -m streamlit run app\\streamlit_app.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

# Streamlit puts the app/ folder on the import path, not the project root,
# so add the project root ourselves. Without this, "import src..." fails.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import streamlit as st
import torch

from src.inference.script_generator import format_shot_description
from src.inference.shot_segmentation import segment_video
from src.inference.video_analyzer import (
    N_MOVEMENT_FRAMES,
    STATIC_THRESHOLDS,
    classify_movement,
    classify_static,
    decode_all_frames,
    load_movement_model,
    load_static_model,
)
from src.models.static_classifier import DEFAULT_CLIP_CHECKPOINT, ClipPreprocess

# decode_all_frames() loads EVERY frame of the video into memory, so long or
# high-resolution videos can run out of RAM. This limit is a cautious guess,
# not a measured value. Raise it if your machine copes.
MAX_FRAMES = 1200
THUMB_SIZE = 480

st.set_page_config(page_title="CineVision AI", page_icon="🎬", layout="wide")


@st.cache_resource(show_spinner="Loading models (first time only)...")
def get_models():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    static_model = load_static_model(device)
    movement_model = load_movement_model(device)
    preprocess = ClipPreprocess(DEFAULT_CLIP_CHECKPOINT)
    with open(STATIC_THRESHOLDS) as f:
        static_thresholds = json.load(f)
    return device, static_model, movement_model, preprocess, static_thresholds


def count_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return n


def analyze(video_path, progress):
    device, static_model, movement_model, preprocess, thresholds = get_models()

    shots = segment_video(str(video_path))
    all_frames = decode_all_frames(video_path)
    n_frames = len(all_frames)

    results = []
    skipped = 0
    total = max(len(shots), 1)
    for k, shot in enumerate(shots):
        start_f = shot["start_frame"]
        end_f = min(shot["end_frame"], n_frames)
        if end_f <= start_f:
            skipped += 1
            progress.progress((k + 1) / total, text=f"Shot {k + 1} of {len(shots)} skipped")
            continue

        mid_idx = (start_f + end_f) // 2
        rep_frame = all_frames[mid_idx]
        static_preds = classify_static(static_model, preprocess, rep_frame, thresholds, device)

        idx = np.linspace(start_f, end_f - 1, num=N_MOVEMENT_FRAMES, dtype=int)
        shot_frames = [all_frames[i] for i in idx]
        movement_preds = classify_movement(movement_model, preprocess, shot_frames, device)

        thumb = rep_frame.copy()
        thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))

        results.append({
            "shot_index": shot["shot_index"],
            "start_timecode": shot["start_timecode"],
            "end_timecode": shot["end_timecode"],
            "static": static_preds,
            "movement": movement_preds,
            "thumb": thumb,
        })
        progress.progress((k + 1) / total, text=f"Analyzed shot {k + 1} of {len(shots)}")

    del all_frames
    return results, n_frames, skipped


def build_script_text(video_name, shots):
    """Same layout as generate_script() in script_generator.py."""
    lines = [f"# Shot Breakdown: {video_name}", "", "Scene 1"]
    for shot in shots:
        desc = format_shot_description(shot)
        lines.append(
            f"  Shot {shot['shot_index']} "
            f"({shot['start_timecode']}\u2013{shot['end_timecode']}): {desc}"
        )
    lines.append("")
    return "\n".join(lines)


def show_results(res):
    st.subheader(f"Shot-by-shot breakdown: {res['name']}")
    st.caption(
        f"{len(res['shots'])} shot(s) analyzed from {res['n_frames']} frames. "
        "The whole video is treated as one scene."
    )
    if res["skipped"]:
        st.warning(f"{res['skipped']} shot(s) skipped because their frame range was invalid.")

    for shot in res["shots"]:
        st.divider()
        col_img, col_info = st.columns([1, 2])
        with col_img:
            st.image(shot["thumb"], width=360, caption="Frame the static classifier saw")
        with col_info:
            st.markdown(
                f"### Shot {shot['shot_index']}  \n"
                f"{shot['start_timecode']} \u2192 {shot['end_timecode']}"
            )
            for dim, vals in shot["static"].items():
                st.markdown(f"**{dim}:** {', '.join(vals)}")
            st.markdown(f"**Movement (top 2 guesses):** {', '.join(shot['movement'])}")

    st.divider()
    st.subheader("Script text")
    st.code(res["script_text"], language=None)
    st.download_button(
        "Download shot breakdown (.txt)",
        data=res["script_text"],
        file_name=f"{Path(res['name']).stem}_shot_breakdown.txt",
        mime="text/plain",
    )


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("About this demo")
    st.markdown(
        "**Static labels** (frame size, framing, angle, lens, composition, lighting): "
        "frozen CLIP + 7 small trained heads.\n\n"
        "**Movement:** a small LSTM over 16 frames per shot.\n\n"
        "**Shot cuts:** PySceneDetect."
    )
    st.header("Please read: limits")
    st.markdown(
        "- Both models were trained only on ShotQA (professional film frames). "
        "Other footage, like stock clips or phone video, is out of domain, and "
        "predictions may look repetitive or wrong.\n"
        "- On the ShotBench benchmark the average accuracy across 8 categories was "
        "59.83%, and camera movement was the weakest at 45.8%.\n"
        "- Movement shows the model's top 2 ranked guesses, not confident detections. "
        "The two can even contradict each other.\n"
        "- The video is not split into scenes: all shots go under \"Scene 1\".\n"
        f"- Videos over {MAX_FRAMES} frames are refused, because every frame is loaded into memory."
    )

# --------------------------------------------------------------------- main
st.title("🎬 CineVision AI")
st.write("Upload a short video and get a shot-by-shot cinematography breakdown.")

uploaded = st.file_uploader("Upload a video", type=["mp4", "mov", "mkv", "webm", "avi"])

if uploaded is not None:
    st.video(uploaded.getvalue())

    if st.button("Analyze video", type="primary"):
        tmp_dir = Path(tempfile.mkdtemp())
        video_path = tmp_dir / f"upload{Path(uploaded.name).suffix.lower()}"
        try:
            video_path.write_bytes(uploaded.getvalue())

            n = count_frames(video_path)
            if n > MAX_FRAMES:
                st.error(
                    f"This video has about {n} frames, more than the limit of {MAX_FRAMES}. "
                    "Please upload a shorter or lower-resolution clip."
                )
            else:
                progress = st.progress(0.0, text="Starting...")
                shots, n_frames, skipped = analyze(video_path, progress)
                progress.empty()
                st.session_state["results"] = {
                    "name": uploaded.name,
                    "shots": shots,
                    "n_frames": n_frames,
                    "skipped": skipped,
                    "script_text": build_script_text(uploaded.name, shots),
                }
        except Exception as e:
            st.error(f"Analysis failed: {e}")
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

if "results" in st.session_state:
    show_results(st.session_state["results"])