"""
src/training/eval_shotbench.py

Held-out evaluation on ShotBench (Vchitect/ShotBench).

How it scores (same idea as the published baselines, no thresholds):
  - Every question has 4 options (A-D).
  - Each option gets a SCORE from the model's probabilities.
  - The option with the highest score is the model's answer.
  - Combination options like "Aerial, Overhead" are scored as the AVERAGE
    of the model's probabilities for the tags inside the option.
    (This is OUR design choice -- write it in PROJECT_NOTES.md.)
  - If two or more options tie for the top score, the question gets
    fractional credit (1 / number of tied options) if the right answer
    is among them.
  - If none of an option's tags can be matched to our vocabulary, that
    option gets score -1 (it can never win). We count how often this
    happens so it can be reported honestly.

Usage:
    python -m src.training.eval_shotbench --limit 20     (quick smoke test)
    python -m src.training.eval_shotbench                (full run)
"""

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.data.label_encoding import CLASS_VOCAB
from src.data.movement_label_encoding import MOVEMENT_CLASS_VOCAB, parse_raw_label
from src.inference.video_analyzer import (
    load_static_model,
    load_movement_model,
    decode_all_frames,
    N_MOVEMENT_FRAMES,
)
from src.models.static_classifier import ClipPreprocess, DEFAULT_CLIP_CHECKPOINT

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SHOTBENCH_DIR = PROJECT_ROOT / "data" / "raw" / "shotbench"
REPORTS_DIR = PROJECT_ROOT / "reports"

# ShotBench category name (lowercase) -> our head name
CATEGORY_TO_DIM = {
    "shot size": "Frame Size",
    "lens size": "Lens Size",
    "composition": "Composition",
    "shot framing": "Shot Framing",
    "camera angle": "Camera Angle",
    "lighting type": "Lighting Type",
    "lighting": "Lighting",
    "lighting condition": "Lighting",
}
MOVEMENT_CATEGORY = "camera movement"

# Known wording differences: ShotBench text (lowercase) -> our vocab text (lowercase)
ALIASES = {
    "Shot Framing": {"single": "clean single"},
}

NO_MATCH_SCORE = -1.0
BATCH_SIZE = 32


# ---------------------------------------------------------------- parsing
def parse_list_cell(cell):
    """test.tsv stores lists as strings like "['image/ABC.jpg']"."""
    if isinstance(cell, list):
        return cell
    return ast.literal_eval(str(cell))


def parse_options(cell):
    if isinstance(cell, dict):
        return cell
    try:
        return json.loads(cell)
    except Exception:
        return ast.literal_eval(str(cell))


def static_option_tags(text, dim):
    """Returns (list of class indices matched, list of unmatched tokens)."""
    vocab_lower = {c.lower(): i for i, c in enumerate(CLASS_VOCAB[dim])}
    aliases = ALIASES.get(dim, {})

    whole = text.strip().lower()
    whole = aliases.get(whole, whole)
    if whole in vocab_lower:
        return [vocab_lower[whole]], []

    matched, unmatched = [], []
    for part in text.split(","):
        p = part.strip().lower()
        if not p:
            continue
        p = aliases.get(p, p)
        if p in vocab_lower:
            matched.append(vocab_lower[p])
        else:
            unmatched.append(p)
    return matched, unmatched


def movement_option_tags(text):
    tags, _unmatched_clauses = parse_raw_label(text)
    idx = [MOVEMENT_CLASS_VOCAB.index(t) for t in tags]
    return idx, []


def score_option(probs, idx):
    if not idx:
        return NO_MATCH_SCORE
    return float(np.mean([probs[i] for i in idx]))


def pick_answer(scores, answer_pos):
    """Returns (credit, predicted_position, number_of_tied_winners)."""
    scores = np.array(scores, dtype=float)
    best = scores.max()
    winners = [i for i, s in enumerate(scores) if np.isclose(s, best)]
    credit = (1.0 / len(winners)) if answer_pos in winners else 0.0
    return credit, winners[0], len(winners)


# ---------------------------------------------------------------- model runs
def compute_static_probs(paths, model, preprocess, device):
    """Runs every unique image once. Returns {path: {dim: np.array of probs}}."""
    cache = {}
    failed = []
    for start in tqdm(range(0, len(paths), BATCH_SIZE), desc="Images"):
        batch_paths = paths[start:start + BATCH_SIZE]
        tensors, ok_paths = [], []
        for p in batch_paths:
            try:
                img = Image.open(SHOTBENCH_DIR / p).convert("RGB")
                tensors.append(preprocess(img))
                ok_paths.append(p)
            except Exception as e:
                failed.append((p, str(e)))
        if not tensors:
            continue
        pixel_values = torch.stack(tensors).to(device)
        with torch.no_grad():
            outputs = model(pixel_values)
        for j, p in enumerate(ok_paths):
            cache[p] = {
                dim: torch.sigmoid(logits[j]).cpu().numpy()
                for dim, logits in outputs.items()
            }
    return cache, failed


def compute_movement_probs(paths, model, preprocess, device):
    cache = {}
    failed = []
    for p in tqdm(paths, desc="Videos"):
        try:
            frames = decode_all_frames(SHOTBENCH_DIR / p)
            if len(frames) == 0:
                raise ValueError("no frames decoded")
            idx = np.linspace(0, len(frames) - 1, num=N_MOVEMENT_FRAMES, dtype=int)
            pixel_values = torch.stack([preprocess(frames[i]) for i in idx])
            pixel_values = pixel_values.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(pixel_values)
            cache[p] = torch.sigmoid(logits).squeeze(0).cpu().numpy()
        except Exception as e:
            failed.append((p, str(e)))
    return cache, failed


# ---------------------------------------------------------------- main
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None,
                        help="Only use the first N questions per category (smoke test).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(SHOTBENCH_DIR / "test.tsv", sep="\t")
    df["category_norm"] = df["category"].astype(str).str.strip().str.lower().str.replace("_", " ")

    known = set(CATEGORY_TO_DIM) | {MOVEMENT_CATEGORY}
    unknown_cats = sorted(set(df["category_norm"]) - known)
    if unknown_cats:
        print(f"WARNING: unrecognized categories (skipped): {unknown_cats}")
    df = df[df["category_norm"].isin(known)].copy()

    if args.limit:
        df = df.groupby("category_norm").head(args.limit).copy()
    print(f"Questions to score: {len(df)}")

    # single path per question
    df["path_list"] = df["path"].apply(parse_list_cell)
    bad_path = df["path_list"].apply(len) != 1
    if bad_path.any():
        print(f"WARNING: {int(bad_path.sum())} questions do not have exactly 1 file; skipped.")
        df = df[~bad_path].copy()
    df["file"] = df["path_list"].apply(lambda x: x[0])

    # ---- load models
    print("Loading models...")
    static_model = load_static_model(device)
    movement_model = load_movement_model(device)
    preprocess = ClipPreprocess(DEFAULT_CLIP_CHECKPOINT)

    # ---- run models once per unique file
    is_move = df["category_norm"] == MOVEMENT_CATEGORY
    static_files = sorted(df.loc[~is_move, "file"].unique())
    move_files = sorted(df.loc[is_move, "file"].unique())

    static_cache, static_failed = compute_static_probs(static_files, static_model, preprocess, device)
    move_cache, move_failed = compute_movement_probs(move_files, movement_model, preprocess, device)
    n_failed = len(static_failed) + len(move_failed)
    print(f"Files that failed to load: {n_failed}")
    for p, e in (static_failed + move_failed)[:5]:
        print(f"   {p}: {e}")

    # ---- score every question
    rows = []
    unmatched_counter = {}
    skipped_no_file = 0

    for _, r in df.iterrows():
        cat = r["category_norm"]
        options = parse_options(r["options"])
        letters = sorted(options.keys())
        answer_letter = str(r["answer"]).strip().upper()
        if answer_letter not in letters:
            continue
        answer_pos = letters.index(answer_letter)

        scores, tag_counts, no_match_flags = [], [], []
        if cat == MOVEMENT_CATEGORY:
            if r["file"] not in move_cache:
                skipped_no_file += 1
                continue
            probs = move_cache[r["file"]]
            for L in letters:
                idx, _ = movement_option_tags(options[L])
                scores.append(score_option(probs, idx))
                tag_counts.append(len(idx))
                no_match_flags.append(len(idx) == 0)
                if not idx:
                    unmatched_counter.setdefault(cat, Counter())[options[L]] += 1
        else:
            dim = CATEGORY_TO_DIM[cat]
            if r["file"] not in static_cache:
                skipped_no_file += 1
                continue
            probs = static_cache[r["file"]][dim]
            for L in letters:
                idx, unmatched = static_option_tags(options[L], dim)
                scores.append(score_option(probs, idx))
                tag_counts.append(len(idx) + len(unmatched))
                no_match_flags.append(len(idx) == 0)
                for u in unmatched:
                    unmatched_counter.setdefault(cat, Counter())[u] += 1

        credit, pred_pos, n_ties = pick_answer(scores, answer_pos)
        rows.append({
            "index": r["index"],
            "category": cat,
            "correct_credit": credit,
            "predicted": letters[pred_pos],
            "answer": answer_letter,
            "n_tied": n_ties,
            "answer_tag_count": tag_counts[answer_pos],
            "answer_type": "single" if tag_counts[answer_pos] == 1 else "combination",
            "answer_option_unmatched": bool(no_match_flags[answer_pos]),
            "any_option_unmatched": bool(any(no_match_flags)),
        })

    res = pd.DataFrame(rows)
    if res.empty:
        print("No questions were scored. Something is wrong -- paste this output to Claude.")
        return

    # ---- summaries
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    res.to_csv(REPORTS_DIR / "shotbench_predictions.csv", index=False)

    per_cat = res.groupby("category").agg(
        n_questions=("correct_credit", "size"),
        accuracy=("correct_credit", "mean"),
        n_ties=("n_tied", lambda s: int((s > 1).sum())),
        n_answer_unmatched=("answer_option_unmatched", "sum"),
        n_any_unmatched=("any_option_unmatched", "sum"),
    ).reset_index()
    per_cat["accuracy"] = per_cat["accuracy"].round(4)

    split = res.groupby(["category", "answer_type"])["correct_credit"].agg(["size", "mean"]).reset_index()
    split.columns = ["category", "answer_type", "n_questions", "accuracy"]
    split["accuracy"] = split["accuracy"].round(4)

    macro = float(per_cat["accuracy"].mean())
    micro = float(res["correct_credit"].mean())

    per_cat.to_csv(REPORTS_DIR / "shotbench_results.csv", index=False)
    split.to_csv(REPORTS_DIR / "shotbench_single_vs_combo.csv", index=False)
    summary = {
        "limit_per_category": args.limit,
        "n_questions_scored": int(len(res)),
        "macro_average_accuracy_across_categories": round(macro, 4),
        "micro_average_accuracy_all_questions": round(micro, 4),
        "files_failed_to_load": n_failed,
        "questions_skipped_missing_file": skipped_no_file,
        "random_baseline_approx": 0.25,
    }
    with open(REPORTS_DIR / "shotbench_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n=== Accuracy per category ===")
    print(per_cat.to_string(index=False))
    print("\n=== Single-tag vs combination correct answers ===")
    print(split.to_string(index=False))
    print(f"\nAverage across categories (macro): {macro:.4f}")
    print(f"Average over all questions (micro): {micro:.4f}")
    print(f"Questions skipped (file failed): {skipped_no_file}")
    print("Random guessing is about 0.25")

    print("\n=== Option words we could NOT match to our vocabulary (top 15 per category) ===")
    if not unmatched_counter:
        print("None")
    for cat, counter in unmatched_counter.items():
        print(f"[{cat}]")
        for token, n in counter.most_common(15):
            print(f"   {n:4d} x {token!r}")

    print("\nSaved to reports/: shotbench_results.csv, shotbench_single_vs_combo.csv, "
          "shotbench_predictions.csv, shotbench_summary.json")
    if args.limit:
        print("\nNOTE: this was a smoke test (--limit). Do NOT record these numbers.")


if __name__ == "__main__":
    main()