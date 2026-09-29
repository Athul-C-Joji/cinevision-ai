"""
src/script_planner/planner.py -- Script Planner (Module 3)

Script text -> suggested shots for scenes that have not been filmed yet.

This is LLM orchestration, NOT a trained model. The script is split into
scenes (lines starting with INT. / EXT. are treated as scene headings),
each scene is sent to the Gemini API (free tier) together with the Module 1
label vocabulary, and the reply is checked against that vocabulary. Labels
that are not in the vocabulary are dropped and reported. The check covers
spelling only; it says nothing about whether a suggestion is any good.

Needs GEMINI_API_KEY in the .env file (never commit it).
Optional: CINEVISION_PLANNER_MODEL in .env changes the default model.

Usage:
    python -m src.script_planner.planner data\\scripts\\sample_script.txt
    python -m src.script_planner.planner data\\scripts\\sample_script.txt --max-scenes 2
"""

import argparse
import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from src.script_planner.prompt_templates import (
    build_system_prompt,
    build_user_prompt,
    vocab_as_dict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "scripts"

# Which models are free for you is shown at https://aistudio.google.com/rate-limit
# Tried during development:
#   gemini-2.5-flash       -> 404 "no longer available to new users"
#   gemini-3.8-flash       -> repeated 503 "high demand" errors (temporary overload)
#   gemini-3.5-flash-lite  -> worked
DEFAULT_MODEL = "gemini-3.5-flash-lite"
# Generous, because some Gemini models spend part of this budget on internal "thinking".
MAX_OUTPUT_TOKENS = 8192

# Temporary errors (429 = quota/rate limit, 503 = model overloaded, 500 = server
# hiccup) are retried with a growing wait: 20s, 40s, 60s.
MAX_RETRIES = 3
RETRY_WAIT_SECONDS = 20
RETRYABLE_MARKERS = ("429", "RESOURCE_EXHAUSTED", "503", "UNAVAILABLE", "500", "INTERNAL")

SCENE_HEADING_RE = re.compile(
    r"^\s*(INT\./EXT\.|INT/EXT\.|INT\.|EXT\.|I/E\.)", re.IGNORECASE
)


# ---------------------------------------------------------------- scenes
def split_scenes(script_text):
    """Returns a list of (heading, scene_text). No headings -> one scene."""
    scenes = []
    heading, lines = None, []

    def flush():
        text = "\n".join(lines).strip()
        if heading is not None or text:
            scenes.append((heading or "(no scene heading)", text))

    for line in script_text.splitlines():
        if SCENE_HEADING_RE.match(line):
            flush()
            heading, lines = line.strip(), []
        else:
            lines.append(line)
    flush()
    return [(h, t) for h, t in scenes if t]


# ---------------------------------------------------------------- LLM call
def extract_json(text):
    """Pulls the JSON object out of the reply, tolerating stray fences or text."""
    text = text.strip()
    text = re.sub(r"^```(?:json)?", "", text).strip()
    text = re.sub(r"```$", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("no JSON object found in the reply")
    return json.loads(text[start:end + 1])


def call_llm(client, model, system_prompt, user_prompt, usage):
    """Returns (reply_text, was_cut_off). Retries on temporary errors (429/503/500)."""
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=MAX_OUTPUT_TOKENS,
        response_mime_type="application/json",
    )
    response = None
    for attempt in range(MAX_RETRIES + 1):
        try:
            response = client.models.generate_content(
                model=model, contents=user_prompt, config=config
            )
            break
        except Exception as e:
            msg = str(e)
            temporary = any(marker in msg for marker in RETRYABLE_MARKERS)
            if temporary and attempt < MAX_RETRIES:
                wait = RETRY_WAIT_SECONDS * (attempt + 1)
                print(f"  Temporary error from the API; waiting {wait}s, then retrying "
                      f"({attempt + 1} of {MAX_RETRIES})...")
                time.sleep(wait)
                continue
            raise

    meta = getattr(response, "usage_metadata", None)
    usage["input"] += getattr(meta, "prompt_token_count", 0) or 0
    usage["output"] += getattr(meta, "candidates_token_count", 0) or 0

    was_cut_off = False
    try:
        was_cut_off = "MAX_TOKENS" in str(response.candidates[0].finish_reason)
    except Exception:
        pass
    return (response.text or ""), was_cut_off


def clean_labels(raw_labels, vocab):
    """Keeps only labels that exactly match the vocabulary (ignoring case)."""
    lookup = {dim: {c.lower(): c for c in classes} for dim, classes in vocab.items()}
    clean, invalid = {}, []
    raw_labels = raw_labels if isinstance(raw_labels, dict) else {}

    for dim in vocab:
        values = raw_labels.get(dim, [])
        if isinstance(values, str):
            values = [values]
        kept = []
        for v in values:
            canonical = lookup[dim].get(str(v).strip().lower())
            if canonical is None:
                invalid.append(f"{dim}: {v}")
            elif canonical not in kept:
                kept.append(canonical)
        clean[dim] = kept

    for dim in raw_labels:
        if dim not in vocab:
            invalid.append(f"unknown dimension: {dim}")
    return clean, invalid


def plan_scene(client, model, vocab, system_prompt, heading, scene_text, max_shots, usage):
    text, was_cut_off = call_llm(
        client, model, system_prompt,
        build_user_prompt(heading, scene_text, max_shots), usage,
    )
    warnings = []
    if was_cut_off:
        warnings.append("reply was cut off (max output tokens); the shot list may be incomplete")
    if not text.strip():
        raise ValueError("the model returned an empty reply")

    data = extract_json(text)
    shots = []
    for i, s in enumerate(data.get("shots", []), start=1):
        labels, invalid = clean_labels(s.get("labels", {}), vocab)
        shots.append({
            "shot_number": s.get("shot_number", i),
            "description": str(s.get("description", "")).strip(),
            "purpose": str(s.get("purpose", "")).strip(),
            "labels": labels,
            "invalid_labels": invalid,
        })
    if not shots:
        warnings.append("the reply contained no shots")
    return {"heading": heading, "shots": shots, "warnings": warnings}


def plan_script(script_text, model, max_scenes, max_shots):
    scenes = split_scenes(script_text)
    if not scenes:
        raise ValueError("the script file has no text")
    total_scenes = len(scenes)
    scenes = scenes[:max_scenes]

    vocab = vocab_as_dict()
    system_prompt = build_system_prompt()
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    usage = {"input": 0, "output": 0}

    results = []
    for n, (heading, text) in enumerate(scenes, start=1):
        print(f"Planning scene {n} of {len(scenes)}: {heading}")
        try:
            results.append(plan_scene(client, model, vocab, system_prompt,
                                      heading, text, max_shots, usage))
        except Exception as e:
            results.append({"heading": heading, "shots": [],
                            "warnings": [f"this scene failed: {e}"]})
    return {
        "model": model,
        "scenes_in_script": total_scenes,
        "scenes_planned": len(scenes),
        "usage": usage,
        "scenes": results,
    }


# ---------------------------------------------------------------- output
def format_plan(plan, script_name):
    lines = [
        f"# Suggested Shot Plan: {script_name}",
        "(LLM-generated suggestions from the script text. Not produced by a trained model.)",
        "",
    ]
    for n, scene in enumerate(plan["scenes"], start=1):
        lines.append(f"Scene {n}: {scene['heading']}")
        for shot in scene["shots"]:
            lines.append(f"  Shot {shot['shot_number']}: {shot['description']}")
            label_bits = [f"{dim}: {', '.join(v)}" for dim, v in shot["labels"].items() if v]
            if label_bits:
                lines.append("    " + " | ".join(label_bits))
            if shot["purpose"]:
                lines.append(f"    Why: {shot['purpose']}")
            if shot["invalid_labels"]:
                lines.append(f"    (dropped labels not in the vocabulary: {'; '.join(shot['invalid_labels'])})")
        for w in scene["warnings"]:
            lines.append(f"  WARNING: {w}")
        lines.append("")
    return "\n".join(lines)


def main():
    # Load .env first, so GEMINI_API_KEY and CINEVISION_PLANNER_MODEL are visible below.
    load_dotenv(PROJECT_ROOT / ".env")

    parser = argparse.ArgumentParser(description="Suggest shots for the scenes of a script.")
    parser.add_argument("script_path", type=str)
    parser.add_argument("--model", type=str,
                        default=os.environ.get("CINEVISION_PLANNER_MODEL", DEFAULT_MODEL))
    parser.add_argument("--max-scenes", type=int, default=8,
                        help="Only plan the first N scenes (each scene is one API request).")
    parser.add_argument("--max-shots", type=int, default=6)
    parser.add_argument("--output", type=str, default=None,
                        help="Output .txt path. Defaults to data/scripts/<name>_plan.txt")
    args = parser.parse_args()

    if not os.environ.get("GEMINI_API_KEY"):
        raise SystemExit("GEMINI_API_KEY was not found. Add it to the .env file in the project root.")

    script_path = Path(args.script_path)
    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    script_text = script_path.read_text(encoding="utf-8", errors="replace")

    plan = plan_script(script_text, args.model, args.max_scenes, args.max_shots)
    if plan["scenes_in_script"] > plan["scenes_planned"]:
        print(f"NOTE: the script has {plan['scenes_in_script']} scenes; only the first "
              f"{plan['scenes_planned']} were planned (use --max-scenes to change this).")

    text = format_plan(plan, script_path.name)

    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_txt = Path(args.output) if args.output else DEFAULT_OUTPUT_DIR / f"{script_path.stem}_plan.txt"
    out_txt.write_text(text, encoding="utf-8")
    out_json = out_txt.with_suffix(".json")
    out_json.write_text(json.dumps(plan, indent=2), encoding="utf-8")

    print()
    print(text)
    print(f"Model: {plan['model']}")
    print(f"Tokens used: {plan['usage']['input']} in, {plan['usage']['output']} out")
    print(f"Saved to: {out_txt} and {out_json}")


if __name__ == "__main__":
    main()