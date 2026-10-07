"""
web/app.py

Flask version of the CineVision AI Script Planner page.

Shows the saved sample plan (data/scripts/sample_script_plan.json) with a real
reference frame and movement clip from ShotQA for each suggested shot. The
references come from src/script_planner/retrieval.py (plain tag matching on
the dataset labels, not a model). This page makes NO Gemini call.

ShotQA is cc-by-nc-nd-4.0 (non-commercial): run this locally only, do not
publish the images or clips online.

Run from the project root:
    python web\\app.py
Then open http://127.0.0.1:5000
"""

import json
import sys
import time
from pathlib import Path

# Make "import src..." work when running "python web\app.py".
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from flask import Flask, abort, render_template, send_from_directory, url_for

from src.script_planner.retrieval import IMAGE_DIR, ReferenceIndex, find_references

PLAN_PATH = PROJECT_ROOT / "data" / "scripts" / "sample_script_plan.json"
PORT = 5000

app = Flask(__name__)


def build_view(plan, refs):
    """Turns the plan and the references into simple lists for the template."""
    scenes = []
    for s_i, scene in enumerate(plan["scenes"]):
        shots = []
        for k, shot in enumerate(scene["shots"]):
            ref = {}
            try:
                ref = refs[s_i][k]
            except (IndexError, TypeError):
                pass
            image = ref.get("image")
            clip = ref.get("clip")

            chips = [
                (dim, ", ".join(vals))
                for dim, vals in shot["labels"].items()
                if vals
            ]
            shots.append({
                "number": shot["shot_number"],
                "description": shot["description"],
                "purpose": shot.get("purpose", ""),
                "chips": chips,
                "image": image,
                "image_url": url_for("media_image", img_id=image["img_id"]) if image else None,
                "clip": clip,
                "clip_url": url_for("media_clip", filename=clip["filename"]) if clip else None,
            })
        scenes.append({"heading": scene["heading"], "shots": shots})
    return scenes


# ---- load everything once, when the server starts ----
if not PLAN_PATH.exists():
    sys.exit(f"Plan file not found: {PLAN_PATH}")

with open(PLAN_PATH, encoding="utf-8") as f:
    PLAN = json.load(f)

_t0 = time.time()
INDEX = ReferenceIndex()
REFS = find_references(PLAN, INDEX)
print(f"Reference index and matching ready in {time.time() - _t0:.1f} s.")

with app.app_context():
    pass


@app.route("/")
def home():
    scenes = build_view(PLAN, REFS)
    n_shots = sum(len(s["shots"]) for s in scenes)
    return render_template(
        "index.html",
        plan=PLAN,
        scenes=scenes,
        n_shots=n_shots,
    )


@app.route("/media/image/<img_id>")
def media_image(img_id):
    if Path(img_id).suffix.lower() not in {".jpg", ".jpeg", ".png"}:
        abort(404)
    return send_from_directory(IMAGE_DIR, img_id)


@app.route("/media/clip/<filename>")
def media_clip(filename):
    if not filename.lower().endswith(".mp4"):
        abort(404)
    # conditional=True lets the browser seek inside the video (range requests)
    return send_from_directory(IMAGE_DIR, filename, conditional=True)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=PORT, debug=False)