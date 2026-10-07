"""
src/script_planner/retrieval.py

Finds real ShotQA examples for the shots suggested by the Script Planner:
  - a reference IMAGE whose tags are similar to the suggested labels
  - a reference movement CLIP that has the suggested camera movement

This is plain tag matching on the dataset's own labels. It is not a model and
it is not evaluated. The reference is "a real frame with similar tags", not a
guarantee that the suggestion is good.

Dataset license: ShotQA is cc-by-nc-nd-4.0 (non-commercial use only). Do not
publish these images or clips online.

Test from the project root:
    python -m src.script_planner.retrieval data\\scripts\\sample_script_plan.json
"""

import argparse
import heapq
import json
import random
import sys
import time
from pathlib import Path

import pandas as pd
from PIL import Image

from src.data.label_encoding import CLASS_VOCAB, SOURCE_FIELD

PROJECT_ROOT = Path(__file__).resolve().parents[2]
META_PATH = PROJECT_ROOT / "data" / "raw" / "meta.jsonl"
IMAGE_DIR = PROJECT_ROOT / "data" / "raw" / "images"
MOVEMENT_LABELS = PROJECT_ROOT / "data" / "processed" / "movement_labels_encoded.csv"
MOVEMENT_SPLITS = PROJECT_ROOT / "data" / "processed" / "movement_splits.csv"

# Columns of movement_labels_encoded.csv that are not movement classes.
NON_MOVEMENT_COLUMNS = {"filename", "label", "parsed_tags"}

TOP_CANDIDATES = 60   # how many best-scoring images to try before giving up


def _tags(value):
    """Raw comma-separated string -> frozenset of stripped tags."""
    if not isinstance(value, str):
        return frozenset()
    return frozenset(t.strip() for t in value.split(",") if t.strip())


class ReferenceIndex:
    """Loads the dataset tags once. Building it takes a few seconds."""

    def __init__(self):
        vocab_sets = {dim: frozenset(v) for dim, v in CLASS_VOCAB.items()}

        # ---- images: one entry per row of meta.jsonl
        self.rows = []   # (img_id, title, year, {dimension: frozenset of tags})
        with open(META_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                img_id = row.get("img_id")
                if not img_id:
                    continue
                tags = {}
                for dim in CLASS_VOCAB:
                    raw = _tags(row.get(SOURCE_FIELD[dim]))
                    tags[dim] = raw & vocab_sets[dim]
                self.rows.append((img_id, row.get("title", ""), row.get("time", ""), tags))

        # ---- movement clips: only clips that were successfully extracted
        labels = pd.read_csv(MOVEMENT_LABELS)
        splits = pd.read_csv(MOVEMENT_SPLITS)
        labels = labels[labels["filename"].isin(set(splits["filename"]))]
        self.move_cols = [c for c in labels.columns if c not in NON_MOVEMENT_COLUMNS]
        self.move_df = labels.reset_index(drop=True)

    # ------------------------------------------------------------ images
    def find_image(self, labels, exclude=(), seed=0):
        """
        labels: {dimension: [label, ...]} using the 7 static dimensions
                (a "Movement" key, if present, is ignored here).
        Returns a dict, or None if nothing usable was found.
        """
        wanted = {
            dim: frozenset(vals)
            for dim, vals in labels.items()
            if dim in CLASS_VOCAB and vals
        }
        if not wanted:
            return None

        rng = random.Random(seed)
        scored = []
        for i, (img_id, _title, _year, tags) in enumerate(self.rows):
            if img_id in exclude:
                continue
            score = 0.0
            for dim, w in wanted.items():
                c = tags[dim]
                inter = len(w & c)
                if inter:
                    score += inter / len(w | c)
            if score > 0:
                scored.append((score, rng.random(), i))

        for score, _tie, i in heapq.nlargest(TOP_CANDIDATES, scored):
            img_id, title, year, tags = self.rows[i]
            path = IMAGE_DIR / img_id
            if not path.exists():
                continue
            try:
                with Image.open(path) as im:
                    im.verify()           # skips broken files
            except Exception:
                continue
            exact = sum(1 for dim, w in wanted.items() if tags[dim] == w)
            return {
                "img_id": img_id,
                "path": str(path),
                "title": title,
                "year": year,
                "exact_dims": exact,
                "n_dims": len(wanted),
                "score": round(score, 3),
            }
        return None

    # ------------------------------------------------------------- clips
    def find_clip(self, movements, exclude=(), seed=0):
        """
        movements: list of movement names, e.g. ["Push in"].
        Returns a dict, or None if the movement is unknown or has no clip.
        """
        wanted = [m for m in movements if m in self.move_cols]
        if not wanted:
            return None

        df = self.move_df
        sub = df[(df[wanted] == 1).all(axis=1)]
        if sub.empty:
            return None
        n_candidates = len(sub)
        sub = sub.assign(extra=sub[self.move_cols].sum(axis=1) - len(wanted))
        # random order first, then stable sort: fewest extra movements wins
        sub = sub.sample(frac=1, random_state=seed).sort_values("extra", kind="stable")

        for _, row in sub.iterrows():
            name = row["filename"]
            if name in exclude:
                continue
            path = IMAGE_DIR / name
            if not path.exists():
                continue
            return {
                "filename": name,
                "path": str(path),
                "tags": str(row["parsed_tags"]),
                "extra_movements": int(row["extra"]),
                "n_candidates": n_candidates,
            }
        return None


def find_references(plan, index):
    """
    plan: the dict returned by plan_script() (or loaded from *_plan.json).
    Returns a list (one item per scene) of lists (one item per shot) of
    {"image": dict or None, "clip": dict or None}.
    The same image or clip is not used twice.
    """
    used_images, used_clips = set(), set()
    out = []
    k = 0
    for scene in plan["scenes"]:
        scene_refs = []
        for shot in scene["shots"]:
            labels = shot.get("labels", {})
            static = {d: v for d, v in labels.items() if d != "Movement"}
            image = index.find_image(static, exclude=used_images, seed=k)
            clip = index.find_clip(labels.get("Movement", []), exclude=used_clips, seed=k)
            if image:
                used_images.add(image["img_id"])
            if clip:
                used_clips.add(clip["filename"])
            scene_refs.append({"image": image, "clip": clip})
            k += 1
        out.append(scene_refs)
    return out


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8")   # film titles can have accents
    except Exception:
        pass

    parser = argparse.ArgumentParser(description="Find reference images/clips for a plan.")
    parser.add_argument("plan_json", help="path to a *_plan.json file")
    args = parser.parse_args()

    with open(args.plan_json, encoding="utf-8") as f:
        plan = json.load(f)

    t0 = time.time()
    index = ReferenceIndex()
    print(f"Index built in {time.time() - t0:.1f} s: "
          f"{len(index.rows)} images, {len(index.move_df)} usable movement clips.\n")

    t0 = time.time()
    refs = find_references(plan, index)
    print(f"Matching took {time.time() - t0:.1f} s.\n")

    for s, (scene, scene_refs) in enumerate(zip(plan["scenes"], refs), start=1):
        print(f"Scene {s}: {scene['heading']}")
        for shot, ref in zip(scene["shots"], scene_refs):
            print(f"  Shot {shot['shot_number']}: {shot['description']}")
            img, clip = ref["image"], ref["clip"]
            if img:
                print(f"    image: {img['img_id']}  from '{img['title']}' ({img['year']})  "
                      f"exact on {img['exact_dims']} of {img['n_dims']} dimensions")
            else:
                print("    image: none found")
            if clip:
                print(f"    clip : {clip['filename']}  tags: {clip['tags']}  "
                      f"({clip['n_candidates']} clips had this movement)")
            else:
                print(f"    clip : none ({shot['labels'].get('Movement', [])})")
        print()


if __name__ == "__main__":
    main()