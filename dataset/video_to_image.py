"""
Extract frames from camera videos for one or more scenes.

Usage: edit the config section below, then `uv run python video_to_image.py`

- INPUT_PATH can be a single scene directory (contains videos/) or a parent
  directory containing multiple scene subdirectories.
- NUM_FRAMES=1 extracts only the first frame → Camera_01.jpg
- NUM_FRAMES>1 extracts evenly spaced frames (first, interior, last)
  → Camera_01_1.jpg, Camera_01_2.jpg, ...
"""

import cv2
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_PATH = "/home/user/thesis/code/dataset/MTMC_Tracking_2025/val/"
NUM_FRAMES = 1   # 1 = first frame only; >1 = evenly spaced including first & last
IMAGE_EXT  = ".jpg"
# ─────────────────────────────────────────────────────────────────────────────


def is_scene_dir(path: Path) -> bool:
    return (path / "videos").is_dir()


def find_scenes(input_path: Path) -> list[Path]:
    if is_scene_dir(input_path):
        return [input_path]
    scenes = [p for p in sorted(input_path.iterdir()) if p.is_dir() and is_scene_dir(p)]
    if not scenes:
        raise ValueError(f"No scenes found under {input_path}")
    return scenes


def frame_indices(total_frames: int, n: int) -> list[int]:
    if n == 1:
        return [0]
    return list(np.linspace(0, total_frames - 1, n, dtype=int))


def extract_frames(video_path: Path, out_dir: Path, cam_name: str, n: int) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"  [WARN] Cannot open {video_path.name}, skipping")
        return

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        print(f"  [WARN] {video_path.name} reports 0 frames, skipping")
        cap.release()
        return

    indices = frame_indices(total, min(n, total))

    for seq, idx in enumerate(indices, start=1):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok:
            print(f"  [WARN] Could not read frame {idx} from {video_path.name}")
            continue

        if n == 1:
            filename = f"{cam_name}{IMAGE_EXT}"
        else:
            filename = f"{cam_name}_{seq}{IMAGE_EXT}"

        cv2.imwrite(str(out_dir / filename), frame)
        print(f"  Saved {filename}  (frame {idx}/{total - 1})")

    cap.release()


def process_scene(scene_dir: Path, n: int) -> None:
    print(f"\nScene: {scene_dir.name}")
    videos_dir = scene_dir / "videos"
    out_dir = scene_dir / "images"
    out_dir.mkdir(exist_ok=True)

    video_files = sorted(videos_dir.glob("*.mp4"))
    if not video_files:
        print("  No .mp4 files found.")
        return

    for vf in video_files:
        cam_name = vf.stem   # e.g. Camera_01
        print(f"  {cam_name}")
        extract_frames(vf, out_dir, cam_name, n)


if __name__ == "__main__":
    input_path = Path(INPUT_PATH)
    if not input_path.exists():
        raise FileNotFoundError(f"Path not found: {input_path}")

    scenes = find_scenes(input_path)
    print(f"Found {len(scenes)} scene(s). Extracting {NUM_FRAMES} frame(s) per camera.")

    for scene in scenes:
        process_scene(scene, NUM_FRAMES)

    print("\nDone.")