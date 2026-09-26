"""
Script Generator (Module 2). Takes Video Analyzer output (per-shot static +
movement predictions) and formats it into a readable shot-by-shot breakdown,
in the style of a shot list / script annotation.

Note: PySceneDetect only detects shot (cut) boundaries, not scene groupings
(i.e. it has no notion of "these 3 shots are the same scene"). True scene
grouping would need separate logic (e.g. clustering by visual similarity or
location continuity) that hasn't been built. For now, the entire video is
treated as a single scene containing all detected shots — this is an
accurate reflection of what the pipeline currently does, not full scene
segmentation.
"""

import argparse
from pathlib import Path

from src.inference.video_analyzer import analyze_video

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "scripts"

# Preferred order + light rewording for natural shot-description phrasing.
# Dimensions not listed here still get included, just appended after.
DESCRIPTION_ORDER = ["Frame Size", "Shot Framing", "Camera Angle", "Lens Size"]
LIGHTING_DIMS = ["Lighting Type", "Lighting"]


def format_shot_description(shot):
    """Turn one shot's static + movement predictions into a readable line."""
    static = shot["static"]
    parts = []

    # Core framing/angle/lens phrase
    for dim in DESCRIPTION_ORDER:
        if dim in static and static[dim]:
            parts.append(" / ".join(static[dim]))
    framing_phrase = ", ".join(parts) if parts else "unclassified framing"

    # Composition (only if present, optional detail)
    composition = static.get("Composition", [])
    composition_phrase = f" Composition: {', '.join(composition)}." if composition else ""

    # Lighting
    lighting_bits = []
    for dim in LIGHTING_DIMS:
        if dim in static and static[dim]:
            lighting_bits.extend(static[dim])
    lighting_phrase = f" Lighting: {', '.join(lighting_bits)}." if lighting_bits else ""

    # Movement
    movement = shot.get("movement", [])
    movement_phrase = f" Movement: {', '.join(movement)}." if movement else ""

    return f"{framing_phrase}.{composition_phrase}{lighting_phrase}{movement_phrase}"


def generate_script(video_path, device=None):
    """
    Run the full Video Analyzer pipeline and format results into a
    shot-by-shot script breakdown. Returns the formatted text as a string.
    """
    shots = analyze_video(video_path, device=device)

    lines = [f"# Shot Breakdown: {Path(video_path).name}", ""]
    lines.append("Scene 1")
    for shot in shots:
        desc = format_shot_description(shot)
        lines.append(
            f"  Shot {shot['shot_index']} "
            f"({shot['start_timecode']}–{shot['end_timecode']}): {desc}"
        )
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Generate a shot-by-shot script breakdown for a video.")
    parser.add_argument("video_path", type=str)
    parser.add_argument("--output", type=str, default=None,
                         help="Output .txt path. Defaults to data/scripts/<video_name>.txt")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    script_text = generate_script(video_path)

    if args.output:
        output_path = Path(args.output)
    else:
        DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        output_path = DEFAULT_OUTPUT_DIR / f"{video_path.stem}.txt"

    output_path.write_text(script_text, encoding="utf-8")

    print(script_text)
    print(f"\nSaved to: {output_path}")


if __name__ == "__main__":
    main()