"""
Floor normal visualization.

Produces a single overlay image:
  floor_normal_overlay.png

  — original RGB surveillance frame (Camera_05, frame 0)
  — temporal floor mask M_floor_acc as a semi-transparent green region
  — derived floor normal n projected back onto the image as a yellow 3D arrow
    originating from the floor centroid and pointing strictly upward into the scene

Imports MetricFloorNormalExtractor from geometry/normals.py unchanged.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import json
import numpy as np

from geometry.normals import MetricFloorNormalExtractor

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CAMERA_ID    = "Camera_16"
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH   = os.path.join(BASE_DIR, f"dataset/MTMC_Tracking_2025/val/Hospital_000/videos/{CAMERA_ID}.mp4")
DEPTH_PATH   = os.path.join(BASE_DIR, f"depth/temporal_depth2/{CAMERA_ID}_temporal_depth_raw.npy")
MASK_NPZ     = os.path.join(BASE_DIR, f"segmentation/temporal_masks2/{CAMERA_ID}_temporal_bg.npz")
CALIB_JSON   = os.path.join(BASE_DIR, "dataset/MTMC_Tracking_2025/val/Hospital_000/calibration.json")
OUTPUT_DIR   = os.path.dirname(os.path.abspath(__file__))

# Visual parameters
MASK_ALPHA   = 0.40   # floor overlay opacity
ARROW_COLOR  = (0, 220, 255)   # bright yellow in BGR
ARROW_THICK  = 4
ARROW_TIP    = 0.25             # arrowhead as fraction of arrow length
NORMAL_SCALE = 1.0              # multiplier on camera_height for arrow length
# ---------------------------------------------------------------------------


def load_K(calib_path: str, camera_id: str) -> np.ndarray:
    with open(calib_path) as f:
        data = json.load(f)
    sensors = [s for s in data["sensors"] if s["type"] == "camera"]
    cam = next(s for s in sensors if s["id"] == camera_id)
    return np.array(cam["intrinsicMatrix"], dtype=np.float64)


def floor_centroid_3d(floor_mask: np.ndarray, depth: np.ndarray, K: np.ndarray):
    """
    Return the 3D camera-space point at the floor centroid.
    Falls back to median floor depth if centroid pixel has zero depth.
    """
    vy, vx = np.where(floor_mask == 255)
    u_c = int(round(float(vx.mean())))
    v_c = int(round(float(vy.mean())))

    z = float(depth[v_c, u_c])
    if z <= 0:
        floor_depths = depth[floor_mask == 255]
        z = float(np.median(floor_depths[floor_depths > 0]))

    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u_c - cx) * z / fx
    y = (v_c - cy) * z / fy
    return np.array([x, y, z]), (u_c, v_c)


def project(P: np.ndarray, K: np.ndarray):
    """Project a 3D camera-space point to integer 2D pixel (u, v)."""
    p = K @ P
    return int(round(p[0] / p[2])), int(round(p[1] / p[2]))


def draw_floor_overlay(frame: np.ndarray, floor_mask: np.ndarray, alpha: float) -> np.ndarray:
    """Blend a semi-transparent green region over the floor pixels."""
    colored = np.zeros_like(frame)
    colored[floor_mask == 255] = [0, 200, 0]   # green (BGR)
    return cv2.addWeighted(frame, 1.0 - alpha, colored, alpha, 0)


def draw_normal_arrow(img: np.ndarray, pt_2d_origin, pt_2d_tip,
                      color, thickness, tip_fraction) -> np.ndarray:
    """Draw a thick arrowed line with an optional dot at the base."""
    out = img.copy()
    cv2.arrowedLine(out, pt_2d_origin, pt_2d_tip, color, thickness,
                    cv2.LINE_AA, tipLength=tip_fraction)
    # Dot at the origin to anchor the arrow visually
    cv2.circle(out, pt_2d_origin, thickness * 2, color, -1, cv2.LINE_AA)
    return out


def main():
    # --- Load data ---
    print(f"Loading frame 0 from {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    assert ret, "Could not read first frame"

    print(f"Loading depth map: {DEPTH_PATH}")
    depth = np.load(DEPTH_PATH)

    print(f"Loading floor mask: {MASK_NPZ}")
    floor_mask = np.load(MASK_NPZ)["floor"]   # uint8, 0/255

    print(f"Loading K matrix from {CALIB_JSON}")
    K = load_K(CALIB_JSON, CAMERA_ID)

    # --- Extract floor normal ---
    print("Running MetricFloorNormalExtractor...")
    extractor = MetricFloorNormalExtractor()
    normal, camera_height = extractor.extract_normal(depth, floor_mask, K)
    assert normal is not None, "Floor normal extraction failed"
    print(f"  normal = {normal}  camera_height = {camera_height:.4f}")

    # --- Compute 3D arrow endpoints ---
    P_origin, (u_c, v_c) = floor_centroid_3d(floor_mask, depth, K)
    P_tip = P_origin + NORMAL_SCALE * camera_height * normal

    pt_origin_2d = project(P_origin, K)
    pt_tip_2d    = project(P_tip, K)
    print(f"  arrow: {pt_origin_2d} → {pt_tip_2d}")

    # --- Compose image ---
    vis = draw_floor_overlay(frame, floor_mask, MASK_ALPHA)
    vis = draw_normal_arrow(vis, pt_origin_2d, pt_tip_2d,
                            ARROW_COLOR, ARROW_THICK, ARROW_TIP)

    out_path = os.path.join(OUTPUT_DIR, "floor_normal_overlay.png")
    cv2.imwrite(out_path, vis)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
