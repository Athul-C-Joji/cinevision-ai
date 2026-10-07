"""
app/streamlit_app.py

CineVision AI demo with three tabs:
  1. Video Analyzer: upload a video, get a shot-by-shot breakdown with a
     summary, a shot timeline, a filmstrip and label-distribution charts.
  2. Script Planner: paste or upload a script, get suggested shots per scene,
     each with a real reference frame and movement clip from ShotQA.
  3. Model results: ShotBench results next to published baselines.

The Video Analyzer tab uses the same building blocks as
src/inference/video_analyzer.py (analyze_video), in the same order, so the app
can also keep a thumbnail per shot and load the models only once. If
analyze_video changes, this file needs the same change.

Difference from analyze_video: instead of decoding EVERY frame into memory,
the app reads the video once, forward only (no seeking), and keeps only the
frames the models need, one shot at a time. The frames chosen are the same
ones analyze_video chooses (the middle frame and 16 evenly spaced frames).

The Script Planner tab calls plan_script() and format_plan() from
src/script_planner/planner.py. It needs GEMINI_API_KEY in the .env file.
The reference frames and clips come from src/script_planner/retrieval.py
(plain tag matching on the ShotQA labels, not a model). ShotQA is
cc-by-nc-nd-4.0, so these images and clips are for local, non-commercial use.

Run from the project root:
    python -m streamlit run app\\streamlit_app.py
"""

import html
import json
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path

# Streamlit puts the app/ folder on the import path, not the project root,
# so add the project root ourselves. Without this, "import src..." fails.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from dotenv import load_dotenv
from PIL import Image

from src.inference.script_generator import format_shot_description
from src.inference.shot_segmentation import segment_video
from src.inference.video_analyzer import (
    N_MOVEMENT_FRAMES,
    STATIC_THRESHOLDS,
    classify_movement,
    classify_static,
    load_movement_model,
    load_static_model,
)
from src.models.static_classifier import DEFAULT_CLIP_CHECKPOINT, ClipPreprocess

# Makes GEMINI_API_KEY (and CINEVISION_PLANNER_MODEL) from .env visible to the
# Script Planner tab. The key is never shown or logged by this app.
load_dotenv(PROJECT_ROOT / ".env")

# The video is read frame by frame (not loaded into memory all at once), so
# this limit is about analysis TIME, not memory. 6000 frames is roughly 4
# minutes at 24 fps. It is a cautious guess, not a measured value. Raise it
# if your machine copes.
MAX_FRAMES = 6000
THUMB_SIZE = 480

MOVEMENT_OPTION = "Movement (top 2 guesses)"

SAMPLE_PLAN_PATH = PROJECT_ROOT / "data" / "scripts" / "sample_script_plan.json"

# Colors for the shot timeline (one color per label, in order of appearance).
TIMELINE_PALETTE = [
    "#E4B363", "#4C9F9F", "#C8553D", "#7B8CDE", "#B07CC6", "#7A9E5B",
    "#E88FB4", "#B5A642", "#9A6B4F", "#58A6E0", "#D98E73", "#A0A0A0",
]

st.set_page_config(page_title="CineVision AI", page_icon="🎬", layout="wide")

# ---------------------------------------------------------------------------
# ShotBench results (all numbers were copied from real evaluation output or
# from the ShotBench project page; the published baselines were NOT re-run).
# Values are accuracy in percent. Row order matches PERF_CATEGORIES.
# "Lighting" here is ShotBench's "Lighting condition".
# ---------------------------------------------------------------------------
PERF_CATEGORIES = [
    "Shot size",
    "Shot framing",
    "Camera angle",
    "Lens size",
    "Lighting type",
    "Composition",
    "Lighting",
    "Camera movement",
    "Average (8 categories)",
]
PERF_TABLE = {
    "CineVision (MLP head)": [84.12, 73.03, 67.25, 52.15, 60.00, 54.28, 57.43, 45.80, 61.76],
    "CineVision (old linear head)": [77.11, 74.16, 65.49, 50.72, 59.26, 53.24, 52.86, 45.80, 59.83],
    "GPT-4o (published)": [69.3, 83.1, 58.2, 48.9, 63.2, 55.2, 48.0, 48.3, 59.3],
    "Qwen2.5-VL-7B (published)": [69.1, 73.5, 53.2, 47.0, 60.5, 49.9, 47.4, 30.2, 53.8],
    "ShotVL-7B (published)": [81.2, 90.1, 78.0, 68.5, 70.1, 45.7, 64.3, 62.9, 70.1],
}


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


def get_fps(video_path):
    """Frames per second of the video, or None if it cannot be read."""
    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return float(fps) if fps and fps > 0 else None


class SequentialFrameReader:
    """
    Reads a video once, forward only, and hands back only the frames asked
    for, as PIL RGB images (the same kind decode_all_frames returns).

    No seeking is used, because OpenCV's seek-based frame access is
    unreliable on some codecs. Frame numbers must be requested in
    increasing order across calls, which holds for shots in time order.
    """

    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(str(video_path))
        self.pos = 0          # number of frames read (grabbed) so far
        self.ended = False

    def get(self, indices):
        """indices: sorted, unique frame numbers. Returns {index: PIL image}.
        Frames the video does not have are simply missing from the result."""
        out = {}
        for idx in indices:
            if idx < self.pos:
                raise ValueError(
                    f"Frame {idx} was requested after the reader passed it. "
                    "Shots must be processed in time order."
                )
            while self.pos < idx and not self.ended:
                if self.cap.grab():
                    self.pos += 1
                else:
                    self.ended = True
            if self.ended:
                break
            if not self.cap.grab():
                self.ended = True
                break
            self.pos += 1                       # this was frame number idx
            ok, frame = self.cap.retrieve()
            if ok:
                out[idx] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        return out

    def drain(self):
        """Read to the end and return the total number of frames."""
        while not self.ended:
            if self.cap.grab():
                self.pos += 1
            else:
                self.ended = True
        return self.pos

    def close(self):
        self.cap.release()


def analyze(video_path, progress):
    device, static_model, movement_model, preprocess, thresholds = get_models()

    fps = get_fps(video_path)
    n_est = count_frames(video_path)   # container's frame count; may be inexact
    shots = segment_video(str(video_path))

    reader = SequentialFrameReader(video_path)
    results = []
    skipped = 0
    total = max(len(shots), 1)
    try:
        for k, shot in enumerate(shots):
            start_f = shot["start_frame"]
            end_f = shot["end_frame"]
            if n_est > 0:
                end_f = min(end_f, n_est)
            if end_f <= start_f:
                skipped += 1
                progress.progress((k + 1) / total, text=f"Shot {k + 1} of {len(shots)} skipped")
                continue

            # Same frames analyze_video uses: the middle frame for the static
            # classifier and N_MOVEMENT_FRAMES evenly spaced frames for movement.
            mid_idx = (start_f + end_f) // 2
            idx = np.linspace(start_f, end_f - 1, num=N_MOVEMENT_FRAMES, dtype=int)
            wanted = sorted({int(mid_idx)} | {int(i) for i in idx})

            got = reader.get(wanted)
            if any(i not in got for i in wanted):
                # The video ended or a frame could not be decoded.
                skipped += 1
                progress.progress((k + 1) / total, text=f"Shot {k + 1} of {len(shots)} skipped")
                continue

            rep_frame = got[int(mid_idx)]
            static_preds = classify_static(static_model, preprocess, rep_frame, thresholds, device)

            shot_frames = [got[int(i)] for i in idx]
            movement_preds = classify_movement(movement_model, preprocess, shot_frames, device)

            thumb = rep_frame.copy()
            thumb.thumbnail((THUMB_SIZE, THUMB_SIZE))

            results.append({
                "shot_index": shot["shot_index"],
                "start_timecode": shot["start_timecode"],
                "end_timecode": shot["end_timecode"],
                "start_frame": start_f,
                "end_frame": end_f,
                "static": static_preds,
                "movement": movement_preds,
                "thumb": thumb,
            })
            del got, shot_frames
            progress.progress((k + 1) / total, text=f"Analyzed shot {k + 1} of {len(shots)}")

        n_frames = reader.drain()
    finally:
        reader.close()

    return results, n_frames, skipped, fps


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


# ---------------------------------------------------------- dashboard parts
def fmt_length(frames, fps):
    """Length as seconds if fps is known, otherwise as a frame count."""
    if fps:
        return f"{frames / fps:.1f} s"
    return f"{int(frames)} frames"


def text_color_for(hex_color):
    """Dark text on light colors, white text on dark colors."""
    r = int(hex_color[1:3], 16)
    g = int(hex_color[3:5], 16)
    b = int(hex_color[5:7], 16)
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "#111111" if luminance > 150 else "#ffffff"


def pick_dim(shots, wanted):
    """Use the wanted dimension name if it exists, else the first one."""
    dims = list(shots[0]["static"].keys())
    if wanted in dims:
        return wanted
    return dims[0] if dims else None


def most_common_label(shots, dim):
    counter = Counter()
    for s in shots:
        counter.update(list(s["static"].get(dim, [])))
    return counter.most_common(1)[0][0] if counter else "-"


def most_common_movement(shots):
    counter = Counter()
    for s in shots:
        counter.update(list(s["movement"]))
    return counter.most_common(1)[0][0] if counter else "-"


def render_summary_tiles(shots, n_frames, fps):
    shot_frames = [s["end_frame"] - s["start_frame"] for s in shots]
    avg_frames = sum(shot_frames) / len(shot_frames)
    dim = pick_dim(shots, "Frame Size")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Shots", len(shots))
    c2.metric("Video length", fmt_length(n_frames, fps))
    c3.metric("Average shot length", fmt_length(avg_frames, fps))
    if dim:
        c4.metric(f"Most common {dim}", most_common_label(shots, dim))
    c5.metric("Most frequent movement guess", most_common_movement(shots))
    st.caption(
        "Movement values are the model's forced top-2 guesses per shot, "
        "not confident detections."
    )


def render_timeline(shots, fps):
    """One colored block per shot, width = duration, color = one label."""
    dim = pick_dim(shots, "Frame Size")
    if dim is None:
        return

    durations = [s["end_frame"] - s["start_frame"] for s in shots]
    total = max(sum(durations), 1)

    color_of = {}
    segments = []
    for s, d in zip(shots, durations):
        labels = list(s["static"].get(dim, []))
        label = labels[0] if labels else "(none)"
        if label not in color_of:
            color_of[label] = TIMELINE_PALETTE[len(color_of) % len(TIMELINE_PALETTE)]
        color = color_of[label]
        width = 100.0 * d / total
        tip = html.escape(
            f"Shot {s['shot_index']}: {label} ({fmt_length(d, fps)})", quote=True
        )
        segments.append(
            f'<div title="{tip}" style="width:{width:.3f}%;background:{color};'
            f"border-right:2px solid #0E1117;display:flex;align-items:center;"
            f"justify-content:center;color:{text_color_for(color)};font-size:12px;"
            f'font-weight:600;overflow:hidden;white-space:nowrap;">{s["shot_index"]}</div>'
        )

    bar = (
        '<div style="display:flex;width:100%;height:46px;border-radius:6px;'
        'overflow:hidden;">' + "".join(segments) + "</div>"
    )
    legend = "".join(
        f'<span style="display:inline-block;margin-right:14px;font-size:13px;">'
        f'<span style="display:inline-block;width:12px;height:12px;'
        f'background:{color};border-radius:2px;margin-right:5px;"></span>'
        f"{html.escape(label)}</span>"
        for label, color in color_of.items()
    )
    st.markdown(bar + '<div style="margin-top:8px;">' + legend + "</div>", unsafe_allow_html=True)
    st.caption(
        f"Block width = shot duration. Color = the first predicted {dim} label. "
        "The number inside a block is the shot number. Hover for details."
    )


def render_distribution(shots):
    dims = list(shots[0]["static"].keys())
    options = dims + [MOVEMENT_OPTION]
    choice = st.selectbox("Dimension", options, key="dist_dim")

    counter = Counter()
    for s in shots:
        labels = s["movement"] if choice == MOVEMENT_OPTION else s["static"].get(choice, [])
        counter.update(list(labels))

    if not counter:
        st.caption("No labels were predicted for this dimension.")
        return

    df = pd.DataFrame({"Shots": pd.Series(counter)}).sort_values("Shots", ascending=False)
    st.bar_chart(df)
    st.caption(
        "Number of shots that received each label (a shot can have several labels). "
        "These are model predictions, not ground truth."
    )


def show_results(res):
    shots = res["shots"]
    fps = res.get("fps")

    st.subheader(f"Shot-by-shot breakdown: {res['name']}")
    st.caption(
        f"{len(shots)} shot(s) analyzed from {res['n_frames']} frames. "
        "The whole video is treated as one scene."
    )
    if res["skipped"]:
        st.warning(
            f"{res['skipped']} shot(s) skipped (invalid frame range, or the video "
            "frames could not be read)."
        )

    if not shots:
        st.info("No shots to show.")
        return

    render_summary_tiles(shots, res["n_frames"], fps)

    st.markdown("#### Shot timeline")
    render_timeline(shots, fps)

    st.markdown("#### Filmstrip")
    st.image(
        [s["thumb"] for s in shots],
        width=150,
        caption=[f"Shot {s['shot_index']}" for s in shots],
    )

    st.markdown("#### Label distribution")
    render_distribution(shots)

    st.markdown("#### Shot details")
    for shot in shots:
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
    st.write("Upload a video and get a shot-by-shot cinematography breakdown.")

    uploaded = st.file_uploader(
        "Upload a video", type=["mp4", "mov", "mkv", "webm", "avi"], key="video_file"
    )

    if uploaded is not None:
        st.video(uploaded.getvalue())

        if st.button("Analyze video", type="primary"):
            # Clear the previous results first, so an old breakdown is never
            # shown under an error about the new video.
            st.session_state.pop("results", None)

            tmp_dir = Path(tempfile.mkdtemp())
            video_path = tmp_dir / f"upload{Path(uploaded.name).suffix.lower()}"
            try:
                video_path.write_bytes(uploaded.getvalue())

                n = count_frames(video_path)
                if n > MAX_FRAMES:
                    st.error(
                        f"This video has about {n} frames, more than the limit of {MAX_FRAMES} "
                        "(a time limit: longer videos take much longer to analyze). "
                        "Please upload a shorter clip."
                    )
                else:
                    progress = st.progress(0.0, text="Detecting shots...")
                    shots, n_frames, skipped, fps = analyze(video_path, progress)
                    progress.empty()
                    st.session_state["results"] = {
                        "name": uploaded.name,
                        "shots": shots,
                        "n_frames": n_frames,
                        "skipped": skipped,
                        "fps": fps,
                        "script_text": build_script_text(uploaded.name, shots),
                    }
            except Exception as e:
                st.error(f"Analysis failed: {e}")
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

    if "results" in st.session_state:
        show_results(st.session_state["results"])


# ================================================================ planner
@st.cache_resource(show_spinner="Loading the reference index (first time only)...")
def get_reference_index():
    # Imported here so a problem with retrieval cannot break the other tabs.
    from src.script_planner.retrieval import ReferenceIndex
    return ReferenceIndex()


def show_shot_references(ref):
    """Shows the reference image and clip for one suggested shot."""
    col_img, col_clip = st.columns(2)

    with col_img:
        img = ref.get("image") if ref else None
        if img:
            st.image(img["path"], width=360)
            st.caption(
                f"Reference frame from \u201c{img['title']}\u201d ({img['year']}). "
                f"Its dataset tags match the suggestion exactly on "
                f"{img['exact_dims']} of {img['n_dims']} dimensions."
            )
        else:
            st.caption("No reference frame found for this combination of labels.")

    with col_clip:
        clip = ref.get("clip") if ref else None
        if clip:
            try:
                st.video(Path(clip["path"]).read_bytes(), format="video/mp4")
                st.caption(
                    f"Reference clip, labeled in the dataset as: {clip['tags']}. "
                    f"({clip['n_candidates']} clips in the dataset have this movement.)"
                )
            except Exception as e:
                st.caption(f"Could not load the reference clip: {e}")
        else:
            st.caption("No reference clip for this movement.")


def show_plan(res):
    plan = res["plan"]
    st.subheader(f"Suggested shot plan: {res['name']}")
    st.caption(
        f"{plan['scenes_planned']} of {plan['scenes_in_script']} scene(s) planned."
    )

    # Find the references once per plan, and keep them with the plan.
    if "refs" not in res:
        try:
            from src.script_planner.retrieval import find_references
            res["refs"] = find_references(plan, get_reference_index())
        except Exception as e:
            res["refs"] = None
            res["refs_error"] = str(e)
    refs = res.get("refs")
    if res.get("refs_error"):
        st.warning(f"Reference frames and clips are unavailable: {res['refs_error']}")

    # Planner warnings and dropped labels are collected here and shown in the
    # collapsed "How this plan was made" box below, not in the main view.
    notes = []

    for n, scene in enumerate(plan["scenes"], start=1):
        st.divider()
        st.markdown(f"### Scene {n}: {scene['heading']}")
        for w in scene["warnings"]:
            notes.append(f"Scene {n}: {w}")
        for j, shot in enumerate(scene["shots"]):
            st.markdown(f"**Shot {shot['shot_number']}:** {shot['description']}")
            for dim, vals in shot["labels"].items():
                if vals:
                    st.markdown(f"- **{dim}:** {', '.join(vals)}")
            if shot["purpose"]:
                st.caption(f"Why: {shot['purpose']}")
            if shot["invalid_labels"]:
                notes.append(
                    f"Scene {n}, shot {shot['shot_number']}: dropped (not in the "
                    "label vocabulary): " + "; ".join(shot["invalid_labels"])
                )
            if refs:
                try:
                    show_shot_references(refs[n - 1][j])
                except (IndexError, KeyError):
                    pass

    st.divider()
    with st.expander("How this plan was made", expanded=False):
        st.markdown(
            f"- The shot suggestions come from an LLM (model: {plan['model']}, "
            f"{plan['usage']['input']} tokens in, {plan['usage']['output']} out). "
            "This is prompting, not a trained model, and it has not been evaluated. "
            "Results differ from run to run, and each scene is planned on its own.\n"
            "- Labels are only checked for spelling against the label vocabulary.\n"
            "- The reference frames and clips are real examples from the ShotQA "
            "dataset, picked by plain tag matching on the suggested camera labels. "
            "They show the camera style, not your story.\n"
            "- ShotQA is non-commercial (cc-by-nc-nd-4.0): use these locally only."
        )
        if notes:
            st.markdown("**Notes from the planner:**")
            for note in notes:
                st.markdown(f"- {note}")

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
        "scene, using the same labels as the Video Analyzer, with a real "
        "reference frame and clip from the dataset for each suggestion."
    )

    # Shows a plan that was saved earlier. No API call, no quota used.
    if SAMPLE_PLAN_PATH.exists():
        if st.button("Load saved sample plan (no API call)", key="load_sample_plan"):
            try:
                with open(SAMPLE_PLAN_PATH, encoding="utf-8") as f:
                    saved_plan = json.load(f)
                st.session_state["plan"] = {
                    "name": "sample_script.txt",
                    "plan": saved_plan,
                    "text": format_plan(saved_plan, "sample_script.txt"),
                }
            except Exception as e:
                st.error(f"Could not load the saved plan: {e}")

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

    st.caption("\u201cPlan shots\u201d sends the script text to the Gemini API.")
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


# ============================================================ model results
def render_results_tab():
    st.write(
        "Accuracy on the ShotBench benchmark (3,572 multiple-choice questions, "
        "4 options each, so random guessing is about 25%), next to the published "
        "baselines. Higher is better."
    )

    df = pd.DataFrame(PERF_TABLE, index=PERF_CATEGORIES)
    column_cfg = {
        name: st.column_config.ProgressColumn(
            name, min_value=0, max_value=100, format="%.1f"
        )
        for name in df.columns
    }
    st.dataframe(df, column_config=column_cfg)

    st.markdown("#### CineVision (MLP head) by category")
    st.bar_chart(df[["CineVision (MLP head)"]].drop(index="Average (8 categories)"))

    with st.expander("How to read this", expanded=False):
        st.markdown(
            "- The current static model (frozen CLIP ViT-B/32 + MLP head) averages "
            "61.76% over the 8 categories. It is **comparable to** GPT-4o's published "
            "59.3% average, not better: the scoring methods differ, and the rough "
            "error margin is about \u00b15 points per category.\n"
            "- The published baselines come from the ShotBench project page and were "
            "**not re-run** here. They answer from the question text; this model "
            "scores each answer option from its predicted label probabilities.\n"
            "- ShotVL-7B is a fully fine-tuned vision-language model. This system is "
            "a frozen CLIP with small trained heads, so a gap to ShotVL-7B is expected.\n"
            "- \"Lighting\" here is ShotBench's \"Lighting condition\".\n"
            "- Camera movement is identical in both CineVision rows because the "
            "movement LSTM did not change.\n"
            "- Overlap with ShotQA (the training data) from the same films but "
            "different scenes cannot be ruled out.\n"
            "- Differences of a few points between rows are within noise. Only "
            "large gaps should be read as real."
        )


# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("About this demo")
    st.markdown(
        "**Static labels** (frame size, framing, angle, lens, composition, lighting): "
        "frozen CLIP ViT-B/32 + 7 small trained heads (MLP head by default).\n\n"
        "**Movement:** a small LSTM over 16 frames per shot.\n\n"
        "**Shot cuts:** PySceneDetect.\n\n"
        "**Script Planner:** prompts the Gemini API. It is not a trained model. "
        "Its reference frames and clips come from plain tag matching on the "
        "ShotQA labels, not from a model."
    )
    with st.expander("Limits of this demo", expanded=False):
        st.markdown(
            "- Both models were trained only on ShotQA (professional film frames). "
            "Other footage, like stock clips or phone video, is out of domain, and "
            "predictions may look repetitive or wrong.\n"
            "- On the ShotBench benchmark (3,572 questions, 8 categories) the current "
            "static model averaged 61.76%, and camera movement was the weakest at 45.8%. "
            "This is comparable to GPT-4o's published 59.3% average, but the scoring "
            "methods differ. See the Model results tab.\n"
            "- Movement shows the model's top 2 ranked guesses, not confident detections. "
            "The two can even contradict each other.\n"
            "- The video is not split into scenes: all shots go under \"Scene 1\".\n"
            f"- Videos over {MAX_FRAMES} frames are refused, because analysis time grows "
            "with video length.\n"
            "- The Script Planner has no accuracy number. Its labels are only checked "
            "for spelling against the vocabulary. The Camera Angle vocabulary has no "
            "eye-level class.\n"
            "- Script Planner reference frames match camera labels only, not story content."
        )

# --------------------------------------------------------------------- main
st.title("🎬 CineVision AI")

tab_analyzer, tab_planner, tab_results = st.tabs(
    ["Video Analyzer", "Script Planner", "Model results"]
)
with tab_analyzer:
    render_analyzer_tab()
with tab_planner:
    render_planner_tab()
with tab_results:
    render_results_tab()