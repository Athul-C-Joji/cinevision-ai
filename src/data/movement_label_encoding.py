"""
src/data/movement_label_encoding.py

Normalizes raw free-text camera movement labels (from data/processed/movement_labels.csv)
into a clean multi-hot encoded vocabulary of atomic camera movements.

Mirrors the CLASS_VOCAB + encode_multihot() pattern from src/data/label_encoding.py,
but starts from raw text instead of a clean tag field, so it first parses "and" /
"Firstly X, then Y" compound descriptions into individual movement mentions.

Usage:
    python -m src.data.movement_label_encoding
"""

import re
import pandas as pd
from pathlib import Path

MOVEMENT_CLASS_VOCAB = [
    "Push in", "Push out",
    "Pull in", "Pull out",
    "Tilt up", "Tilt down",
    "Pan left", "Pan right",
    "Boom up", "Boom down",
    "Tracking",
    "Trucking left", "Trucking right",
    "Move left", "Move right",
    "Arc",
    "Camera roll",
    "Rack focus",
    "Zoom in", "Zoom out",
    "Static shot",
]

_CLAUSE_SPLIT_RE = re.compile(
    r"\band\b|\bthen\b|\bfirstly\b|\bsecondly\b|\bfinally\b|,|;|\.",
    re.IGNORECASE,
)


def _split_clauses(raw_label: str) -> list[str]:
    parts = _CLAUSE_SPLIT_RE.split(raw_label)
    return [p.strip().lower() for p in parts if p.strip()]


def _classify_clause(clause: str) -> list[str]:
    found = []

    def has(*words):
        return all(re.search(rf"\b{w}\b", clause) for w in words)

    if has("push"):
        if has("out"):
            found.append("Push out")
        elif has("in"):
            found.append("Push in")
    if has("pull"):
        if has("in"):
            found.append("Pull in")
        elif has("out"):
            found.append("Pull out")
    if has("tilt"):
        if has("up"):
            found.append("Tilt up")
        elif has("down"):
            found.append("Tilt down")
    if has("pan"):
        if has("left"):
            found.append("Pan left")
        elif has("right"):
            found.append("Pan right")
    if has("boom"):
        if has("up"):
            found.append("Boom up")
        elif has("down"):
            found.append("Boom down")
    if has("truck") or has("trucking"):
        if has("left"):
            found.append("Trucking left")
        elif has("right"):
            found.append("Trucking right")
    elif has("track") or has("tracking"):
        found.append("Tracking")
    if has("move") or has("moving"):
        if has("left"):
            found.append("Move left")
        elif has("right"):
            found.append("Move right")
    if has("arc"):
        found.append("Arc")
    if has("roll"):
        found.append("Camera roll")
    if has("rack") and has("focus"):
        found.append("Rack focus")
    if has("zoom"):
        if has("in"):
            found.append("Zoom in")
        elif has("out"):
            found.append("Zoom out")
    if has("static"):
        found.append("Static shot")

    return found


def parse_raw_label(raw_label: str) -> tuple[list[str], list[str]]:
    tags: list[str] = []
    unmatched: list[str] = []
    for clause in _split_clauses(raw_label):
        clause_tags = _classify_clause(clause)
        if clause_tags:
            tags.extend(clause_tags)
        else:
            unmatched.append(clause)
    seen = set()
    deduped = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped, unmatched


def encode_multihot(tags: list[str]) -> list[int]:
    return [1 if cls in tags else 0 for cls in MOVEMENT_CLASS_VOCAB]


def main():
    project_root = Path(__file__).resolve().parents[2]
    in_path = project_root / "data" / "processed" / "movement_labels.csv"
    out_path = project_root / "data" / "processed" / "movement_labels_encoded.csv"
    report_path = project_root / "reports" / "movement_label_unmatched.csv"

    df = pd.read_csv(in_path)

    label_col = "label"

    if label_col not in df.columns:
        raise ValueError(
            f"Column '{label_col}' not found in {in_path}. "
            f"Available columns: {list(df.columns)}"
        )

    all_tags = []
    unmatched_rows = []

    for idx, raw in df[label_col].items():
        raw_str = "" if pd.isna(raw) else str(raw)
        tags, unmatched = parse_raw_label(raw_str)
        all_tags.append(tags)
        for clause in unmatched:
            unmatched_rows.append({
                "row_index": idx,
                "raw_label": raw_str,
                "unmatched_clause": clause,
            })
        if not tags:
            unmatched_rows.append({
                "row_index": idx,
                "raw_label": raw_str,
                "unmatched_clause": "(NO TAGS MATCHED AT ALL)",
            })

    multihot = [encode_multihot(t) for t in all_tags]
    for i, cls in enumerate(MOVEMENT_CLASS_VOCAB):
        df[cls] = [row[i] for row in multihot]

    df["parsed_tags"] = ["; ".join(t) for t in all_tags]
    df.to_csv(out_path, index=False)

    report_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(unmatched_rows).to_csv(report_path, index=False)

    n_total = len(df)
    n_fully_unmatched = sum(1 for t in all_tags if not t)
    n_partial_unmatched = sum(
        1 for row in unmatched_rows
        if row["unmatched_clause"] != "(NO TAGS MATCHED AT ALL)"
    )

    print(f"Processed {n_total} rows.")
    print(f"Encoded labels written to: {out_path}")
    print(f"Rows with ZERO tags matched: {n_fully_unmatched}")
    print(f"Individual unmatched clauses (partial matches): {n_partial_unmatched}")
    print(f"Full unmatched report written to: {report_path}")
    print()
    print("Review the report CSV -- if you see repeated patterns there, "
          "tell me what they look like and I'll extend the parser.")


if __name__ == "__main__":
    main()