"""
Shot segmentation using PySceneDetect. Splits a video into individual shots
(scenes) by detecting cuts, returning start/end timestamps and frame numbers
for each shot. Module 2 (Script Generator) will call segment_video() to get
per-shot boundaries, then run each shot through the Module 1 static
classifier and the movement LSTM.
"""

import argparse
from pathlib import Path

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


def segment_video(video_path: str, threshold: float = 27.0):
    """
    Detect scene/shot boundaries in a video.

    Args:
        video_path: path to the video file.
        threshold: ContentDetector sensitivity — lower = more sensitive
            (more cuts detected), higher = less sensitive. 27.0 is
            PySceneDetect's standard default, good starting point.

    Returns:
        List of (shot_index, start_timecode, end_timecode,
                 start_frame, end_frame) tuples, one per detected shot.
    """
    video = open_video(video_path)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=threshold))

    scene_manager.detect_scenes(video)
    scene_list = scene_manager.get_scene_list(start_in_scene=True)

    shots = []
    for i, (start, end) in enumerate(scene_list):
        shots.append({
            "shot_index": i + 1,
            "start_timecode": start.get_timecode(),
            "end_timecode": end.get_timecode(),
            "start_frame": start.frame_num,
            "end_frame": end.frame_num,
        })

    return shots


def main():
    parser = argparse.ArgumentParser(description="Detect shot boundaries in a video.")
    parser.add_argument("video_path", type=str, help="Path to the video file")
    parser.add_argument("--threshold", type=float, default=27.0,
                         help="ContentDetector sensitivity (default: 27.0)")
    args = parser.parse_args()

    video_path = Path(args.video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    print(f"Analyzing: {video_path}")
    shots = segment_video(str(video_path), threshold=args.threshold)

    print(f"\nDetected {len(shots)} shot(s):\n")
    for shot in shots:
        print(f"  Shot {shot['shot_index']}: "
              f"{shot['start_timecode']} -> {shot['end_timecode']} "
              f"(frames {shot['start_frame']}-{shot['end_frame']})")


if __name__ == "__main__":
    main()