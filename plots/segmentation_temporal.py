"""
Segmentation temporal aggregation visualization.

Produces five panel images for a chosen camera:
  seg_a_raw_frame.png         — raw RGB frame with heavy occlusion
  seg_b_single_frame_mask.png — Mask2Former output on that one frame (shows holes)
  seg_c_temporal_mask.png     — temporally aggregated mask (clean, hole-free)
  seg_d_last_frame.png        — last frame of the video (raw RGB)
  seg_e_last_frame_mask.png   — Mask2Former output on the last frame

All images are saved to plots/ next to this script.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from PIL import Image

from segmentation.mask2former_video import TemporalVideoSegmenter, create_temporal_annotation

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CAMERA_ID       = "Camera_01"
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH      = os.path.join(BASE_DIR, f"dataset/MTMC_Tracking_2025/val/Hospital_000/videos/{CAMERA_ID}.mp4")
MASK_NPZ_PATH   = os.path.join(BASE_DIR, f"segmentation/temporal_masks2/{CAMERA_ID}_temporal_bg.npz")
OUTPUT_DIR      = os.path.dirname(os.path.abspath(__file__))   # plots/
N_SCAN_FRAMES   = 20   # number of frames to scan when searching for occlusion
# ---------------------------------------------------------------------------


def segment_frame(segmenter: TemporalVideoSegmenter, bgr_frame: np.ndarray):
    """Run Mask2Former on a single BGR frame; return (floor, wall, door) uint8 masks."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = segmenter.processor(images=pil, return_tensors="pt").to(segmenter.device)
    with torch.no_grad():
        outputs = segmenter.model(**inputs)
    pred = segmenter.processor.post_process_semantic_segmentation(
        outputs, target_sizes=[pil.size[::-1]]
    )[0].cpu().numpy()
    floor = np.where(pred == segmenter.FLOOR_CLASS_INDEX, 255, 0).astype(np.uint8)
    wall  = np.where(pred == segmenter.WALL_CLASS_INDEX,  255, 0).astype(np.uint8)
    door  = np.where(pred == segmenter.DOOR_CLASS_INDEX,  255, 0).astype(np.uint8)
    return floor, wall, door


def find_most_occluded_frame(video_path: str, temporal_floor: np.ndarray,
                              segmenter: TemporalVideoSegmenter, n_frames: int):
    """
    Sample n_frames evenly through the video; return the BGR frame and its
    single-frame masks where the floor occlusion (hole) is largest.
    Score = fraction of temporal-floor pixels not labelled floor in the single frame.
    """
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.linspace(0, total - 1, n_frames, dtype=int)

    best_frame, best_masks, best_score = None, None, -1.0
    temporal_floor_count = max(np.count_nonzero(temporal_floor), 1)

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue
        floor, wall, door = segment_frame(segmenter, frame)
        # Pixels the accumulated mask says are floor but this single frame misses
        hole = np.count_nonzero((temporal_floor == 255) & (floor != 255))
        score = hole / temporal_floor_count
        print(f"  frame {idx:5d}  hole fraction = {score:.3f}")
        if score > best_score:
            best_score = score
            best_frame = frame.copy()
            best_masks = (floor, wall, door)

    cap.release()
    return best_frame, best_masks, best_score


def load_last_frame(video_path: str):
    """Return the last BGR frame of the video."""
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
    ret, frame = cap.read()
    cap.release()
    assert ret, f"Could not read last frame from {video_path}"
    return frame


def main():
    print(f"Loading temporal mask from {MASK_NPZ_PATH}")
    npz = np.load(MASK_NPZ_PATH)
    temporal_floor = npz["floor"]
    temporal_wall  = npz["wall"]
    temporal_door  = npz["door"]

    print("Loading Mask2Former model...")
    segmenter = TemporalVideoSegmenter()

    print(f"Scanning {N_SCAN_FRAMES} frames in {VIDEO_PATH} for heaviest occlusion...")
    best_frame, (sf_floor, sf_wall, sf_door), score = find_most_occluded_frame(
        VIDEO_PATH, temporal_floor, segmenter, N_SCAN_FRAMES
    )

    print(f"Best frame occlusion score: {score:.3f}")

    # --- Panel (a): raw RGB frame with heavy occlusion ---
    out_a = os.path.join(OUTPUT_DIR, "seg_a_raw_frame.png")
    cv2.imwrite(out_a, best_frame)
    print(f"Saved {out_a}")

    # --- Panel (b): single-frame Mask2Former output on that frame (shows holes) ---
    panel_b = create_temporal_annotation(best_frame, sf_floor, sf_wall, sf_door)
    out_b = os.path.join(OUTPUT_DIR, "seg_b_single_frame_mask.png")
    cv2.imwrite(out_b, panel_b)
    print(f"Saved {out_b}")

    # --- Panel (c): temporally aggregated mask overlaid on same frame ---
    panel_c = create_temporal_annotation(best_frame, temporal_floor, temporal_wall, temporal_door)
    out_c = os.path.join(OUTPUT_DIR, "seg_c_temporal_mask.png")
    cv2.imwrite(out_c, panel_c)
    print(f"Saved {out_c}")

    # --- Panel (d): last frame raw RGB ---
    print(f"Loading last frame from {VIDEO_PATH}")
    last_frame = load_last_frame(VIDEO_PATH)
    out_d = os.path.join(OUTPUT_DIR, "seg_d_last_frame.png")
    cv2.imwrite(out_d, last_frame)
    print(f"Saved {out_d}")

    # --- Panel (e): Mask2Former output on last frame ---
    print("Segmenting last frame...")
    lf_floor, lf_wall, lf_door = segment_frame(segmenter, last_frame)
    panel_e = create_temporal_annotation(last_frame, lf_floor, lf_wall, lf_door)
    out_e = os.path.join(OUTPUT_DIR, "seg_e_last_frame_mask.png")
    cv2.imwrite(out_e, panel_e)
    print(f"Saved {out_e}")

    print("Done.")


if __name__ == "__main__":
    main()
