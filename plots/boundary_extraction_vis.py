"""
Boundary extraction visualization.

Produces three panel images for a chosen camera:
  boundary_a_pixels.png     — binary M_boundary mask (thin wall-floor intersection pixels)
  boundary_b_raw_hough.png  — raw PHT line segments overlaid in orange (messy/fragmented)
  boundary_c_merged.png     — merged collinear lines overlaid in cyan (clean vectors)

All images are saved to plots/ next to this script.
Input: pre-computed temporal mask JPG from segmentation/temporal_masks2/.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np

from segmentation.boundary_extraction import merge_collinear_lines

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CAMERA_ID     = "Camera_01"   # 16 raw → 4 merged gives the clearest contrast
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MASK_JPG_PATH = os.path.join(BASE_DIR, f"segmentation/temporal_masks2/{CAMERA_ID}_temporal_bg.jpg")
VIDEO_PATH    = os.path.join(BASE_DIR, f"dataset/MTMC_Tracking_2025/val/Hospital_000/videos/{CAMERA_ID}.mp4")
OUTPUT_DIR    = os.path.dirname(os.path.abspath(__file__))   # plots/
# Hough parameters — mirrors boundary_extraction.py exactly
HOUGH_THRESHOLD   = 50
HOUGH_MIN_LENGTH  = 80
HOUGH_MAX_GAP     = 50
MERGE_ANGLE_DEG   = 5
MERGE_DIST_PX     = 30
# ---------------------------------------------------------------------------


def extract_boundary_pixels(img: np.ndarray):
    """Replicate the HSV-based boundary extraction from boundary_extraction.py."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Floor — green
    floor_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))

    # Wall — red (wraps at hue=0/180)
    wall_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,   50, 50]), np.array([10,  255, 255])),
        cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255])),
    )

    # Dilate floor into wall region → boundary strip
    kernel = np.ones((5, 5), np.uint8)
    floor_dilated = cv2.dilate(floor_mask, kernel, iterations=1)
    boundary = cv2.bitwise_and(floor_dilated, wall_mask)
    return boundary


def draw_segments(base: np.ndarray, lines, color: tuple, thickness: int) -> np.ndarray:
    """Draw a list of [[x1,y1,x2,y2]] segments onto a copy of base."""
    out = base.copy()
    if lines is None:
        return out
    for seg in lines:
        # raw HoughLinesP returns shape (N,1,4); merged returns list of [x1,y1,x2,y2]
        pts = seg[0] if hasattr(seg[0], '__len__') and len(seg[0]) == 4 else seg
        x1, y1, x2, y2 = int(pts[0]), int(pts[1]), int(pts[2]), int(pts[3])
        cv2.line(out, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)
    return out


def load_first_frame(video_path: str) -> np.ndarray:
    """Return the first readable frame from the video as a BGR image."""
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise RuntimeError(f"Could not read frame from {video_path}")
    return frame


def dim_frame(frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Darken a BGR frame so overlaid lines are clearly visible."""
    return (frame.astype(np.float32) * alpha).clip(0, 255).astype(np.uint8)


def main():
    print(f"Loading temporal mask: {MASK_JPG_PATH}")
    mask_img = cv2.imread(MASK_JPG_PATH)
    if mask_img is None:
        raise FileNotFoundError(f"Could not load {MASK_JPG_PATH}")

    print(f"Loading first video frame: {VIDEO_PATH}")
    raw_frame = load_first_frame(VIDEO_PATH)
    bg = dim_frame(raw_frame)   # darkened real frame as neutral background

    # --- Step 1: boundary pixels (derived from temporal mask) ---
    boundary_pixels = extract_boundary_pixels(mask_img)

    # --- Step 2: raw Probabilistic Hough Transform ---
    raw_lines = cv2.HoughLinesP(
        boundary_pixels,
        rho=1,
        theta=np.pi / 180,
        threshold=HOUGH_THRESHOLD,
        minLineLength=HOUGH_MIN_LENGTH,
        maxLineGap=HOUGH_MAX_GAP,
    )
    n_raw = len(raw_lines) if raw_lines is not None else 0
    print(f"Raw PHT segments: {n_raw}")

    # --- Step 3: merge collinear lines ---
    merged_lines = merge_collinear_lines(
        raw_lines,
        angle_tolerance_deg=MERGE_ANGLE_DEG,
        distance_tolerance_px=MERGE_DIST_PX,
    )
    print(f"Merged line segments: {len(merged_lines)}")

    # --- Panel (a): binary boundary-pixel mask on black background ---
    out_a = os.path.join(OUTPUT_DIR, "boundary_a_pixels.png")
    cv2.imwrite(out_a, boundary_pixels)
    print(f"Saved {out_a}")

    # --- Panel (b): raw PHT segments in orange on darkened frame ---
    # Orange (BGR: 0, 165, 255) stands out against any background
    panel_b = draw_segments(bg, raw_lines, color=(0, 165, 255), thickness=2)
    out_b = os.path.join(OUTPUT_DIR, "boundary_b_raw_hough.png")
    cv2.imwrite(out_b, panel_b)
    print(f"Saved {out_b}")

    # --- Panel (c): merged clean lines in bright cyan on darkened frame ---
    panel_c = draw_segments(bg, merged_lines, color=(255, 255, 0), thickness=4)
    out_c = os.path.join(OUTPUT_DIR, "boundary_c_merged.png")
    cv2.imwrite(out_c, panel_c)
    print(f"Saved {out_c}")

    print("Done.")


if __name__ == "__main__":
    main()
