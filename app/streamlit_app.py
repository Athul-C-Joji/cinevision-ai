"""
app/streamlit_app.py

CineVision AI demo with two tabs:
  1. Video Analyzer: upload a video, get a shot-by-shot breakdown.
  2. Script Planner: paste or upload a script, get suggested shots per scene.

The Video Analyzer tab uses the same building blocks as
src/inference/video_analyzer.py (analyze_video), in the same order, so the app
can also keep a thumbnail per shot and load the models only once. If
analyze_video changes, this file needs the same change.

The Script Planner tab calls plan_script() and format_plan() from
src/script_planner/planner.py. It needs GEMINI_API_KEY in the .env file.

Run from the project root:
    python -m streamlit run app\\streamlit_app.py
"""

import json
import os
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
from dotenv import load_dotenv

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

# Makes GEMINI_API_KEY (and CINEVISION_PLANNER_MODEL) from .env visible to the
# Script Planner tab. The key is never shown or logged by this app.
load_dotenv(PROJECT_ROOT / ".env")

# decode_all_frames() loads EVERY frame of the video into memory, so long or
# high-resolution videos can run out of RAM. This limit is a cautious guess,
# not a measured value. Raise it if your machine copes.
MAX_FRAMES = 1200
THUMB_SIZE = 480

st.set_page_config(page_title="CineVision AI", page_icon="🎬", layout="wide")


# =============================================================== analyzer
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


def render_analyzer_tab():
    st.write("Upload a short video and get a shot-by-shot cinematography breakdown.")

    uploaded = st.file_uploader(
        "Upload a video", type=["mp4", "mov", "mkv", "webm", "avi"], key="video_file"
    )

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


# ================================================================ planner
def show_plan(res):
    plan = res["plan"]
    st.subheader(f"Suggested shot plan: {res['name']}")
    st.caption(
        f"Model: {plan['model']}. Planned {plan['scenes_planned']} of "
        f"{plan['scenes_in_script']} scene(s). "
        f"Tokens used: {plan['usage']['input']} in, {plan['usage']['output']} out."
    )

    for n, scene in enumerate(plan["scenes"], start=1):
        st.divider()
        st.markdown(f"### Scene {n}: {scene['heading']}")
        for w in scene["warnings"]:
            st.warning(w)
        for shot in scene["shots"]:
            st.markdown(f"**Shot {shot['shot_number']}:** {shot['description']}")
            for dim, vals in shot["labels"].items():
                if vals:
                    st.markdown(f"- **{dim}:** {', '.join(vals)}")
            if shot["purpose"]:
                st.caption(f"Why: {shot['purpose']}")
            if shot["invalid_labels"]:
                st.caption(
                    "Dropped (not in the label vocabulary): "
                    + "; ".join(shot["invalid_labels"])
                )

    st.divider()
    stem = Path(res["name"]).stem
    col_a, col_b = st.columns(2)
    with col_a:
        st.download_button(
            "Download plan (.txt)",
            data=res["text"],
            file_name=f"{stem}_plan.txt",
            mime="text/plain",
        )
    with col_b:
        st.download_button(
            "Download plan (.json)",
            data=json.dumps(plan, indent=2),
            file_name=f"{stem}_plan.json",
            mime="application/json",
        )


def render_planner_tab():
    # Imported here, so a problem with the planner (for example a missing
    # package) cannot break the Video Analyzer tab.
    try:
        from src.script_planner.planner import (
            DEFAULT_MODEL,
            format_plan,
            plan_script,
            split_scenes,
        )
    except Exception as e:
        st.error(f"Could not load the Script Planner: {e}")
        return

    st.write(
        "Paste a script or upload a .txt file, and get suggested shots for each "
        "scene, using the same labels as the Video Analyzer."
    )
    st.info(
        "This is LLM-generated (Gemini free tier), not a trained model, and it "
        "has not been evaluated. Results differ from run to run. Scenes are "
        "planned one at a time, so the model does not see the other scenes. "
        "Your script text is sent to Google, so use only non-sensitive scripts."
    )

    uploaded_script = st.file_uploader("Upload a script (.txt)", type=["txt"], key="script_file")
    pasted = st.text_area(
        "...or paste script text here (scene headings should start with INT. or EXT.)",
        height=250,
        key="script_text_input",
    )

    script_text, script_name = "", "pasted_script.txt"
    if uploaded_script is not None:
        script_text = uploaded_script.getvalue().decode("utf-8", errors="replace")
        script_name = uploaded_script.name
    elif pasted.strip():
        script_text = pasted

    col1, col2, col3 = st.columns(3)
    with col1:
        max_scenes = st.number_input(
            "Max scenes to plan (1 request each)", min_value=1, max_value=8, value=4, step=1
        )
    with col2:
        max_shots = st.number_input(
            "Max shots per scene", min_value=1, max_value=8, value=6, step=1
        )
    with col3:
        model = st.text_input(
            "Gemini model",
            value=os.environ.get("CINEVISION_PLANNER_MODEL", DEFAULT_MODEL),
            help="Model names and free-tier availability change. See aistudio.google.com/rate-limit",
        )

    if script_text.strip():
        n_scenes = len(split_scenes(script_text))
        st.caption(
            f"Found {n_scenes} scene(s). Will plan the first {min(n_scenes, int(max_scenes))}."
        )

    if st.button("Plan shots", type="primary", key="plan_button"):
        if not script_text.strip():
            st.error("Please upload or paste a script first.")
        elif not os.environ.get("GEMINI_API_KEY"):
            st.error(
                "GEMINI_API_KEY was not found. Add it to the .env file in the project "
                "root, then restart the app. (Never paste the key into the chat or commit it.)"
            )
        else:
            try:
                with st.spinner("Asking the model... this can take a minute, "
                                "and longer if the API is busy and retries."):
                    plan = plan_script(
                        script_text, model.strip(), int(max_scenes), int(max_shots)
                    )
                st.session_state["plan"] = {
                    "name": script_name,
                    "plan": plan,
                    "text": format_plan(plan, script_name),
                }
            except Exception as e:
                st.error(f"Planning failed: {e}")

    if "plan" in st.session_state:
        show_plan(st.session_state["plan"])


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("About this demo")
    st.markdown(
        "**Static labels** (frame size, framing, angle, lens, composition, lighting): "
        "frozen CLIP + 7 small trained heads.\n\n"
        "**Movement:** a small LSTM over 16 frames per shot.\n\n"
        "**Shot cuts:** PySceneDetect.\n\n"
        "**Script Planner:** prompts the Gemini API. It is not a trained model."
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
        f"- Videos over {MAX_FRAMES} frames are refused, because every frame is loaded into memory.\n"
        "- The Script Planner has no accuracy number. Its labels are only checked "
        "for spelling against the vocabulary. The Camera Angle vocabulary has no "
        "eye-level class."
    )

# --------------------------------------------------------------------- main
st.title("🎬 CineVision AI")

tab_analyzer, tab_planner = st.tabs(["Video Analyzer", "Script Planner"])
with tab_analyzer:
    render_analyzer_tab()
with tab_planner:
    render_planner_tab()