"""
Classical corner detection visualization for Camera_12.

Produces a single overlay image:
  corners_overlay.png

  — original RGB surveillance frame
  — each detected wall plane coloured distinctly (semi-transparent)
  — detected 3D room corners projected to 2D and drawn as labelled circles

Imports extract_room_corners, project_3d_to_2d, find_plane_ransac,
unproject_to_3d from geometry/corners_extraction.py unchanged.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import json
import numpy as np
import open3d as o3d

from geometry.corners_extraction import (
    extract_room_corners,
    project_3d_to_2d,
    find_plane_ransac,
    unproject_to_3d,
)

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CAMERA_ID  = "Camera_12"
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH = os.path.join(BASE_DIR, f"dataset/MTMC_Tracking_2025/val/Hospital_000/videos/{CAMERA_ID}.mp4")
DEPTH_PATH = os.path.join(BASE_DIR, f"depth/temporal_depth2/{CAMERA_ID}_temporal_depth_raw.npy")
MASK_NPZ   = os.path.join(BASE_DIR, f"segmentation/temporal_masks2/{CAMERA_ID}_temporal_bg.npz")
CALIB_JSON = os.path.join(BASE_DIR, "dataset/MTMC_Tracking_2025/val/Hospital_000/calibration.json")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# BGR colours per wall (up to 4 walls)
WALL_COLORS = [
    (0,   60,  220),   # red
    (220,  60,    0),  # blue
    (0,  180,  255),   # orange
    (180,   0,  200),  # purple
]
WALL_ALPHA    = 0.50   # wall overlay opacity
CORNER_RADIUS = 14
CORNER_COLOR  = (0, 255, 255)   # bright cyan
CORNER_BORDER = (0, 0, 0)
MAX_WALLS     = 4
DUP_THRESHOLD = 0.95  # dot product — mirrors extract_room_corners
# ---------------------------------------------------------------------------


def load_K(calib_path: str, camera_id: str) -> np.ndarray:
    with open(calib_path) as f:
        data = json.load(f)
    sensors = [s for s in data["sensors"] if s["type"] == "camera"]
    cam = next(s for s in sensors if s["id"] == camera_id)
    return np.array(cam["intrinsicMatrix"], dtype=np.float64)


def extract_walls_with_pixels(depth_map, wall_mask, K):
    """
    Iterative RANSAC wall segmentation, tracking which pixels belong to each
    wall plane.  Mirrors the wall-extraction loop inside extract_room_corners.
    Returns list of (plane_model, row_indices, col_indices).
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # All valid wall pixels
    vy, vx = np.where(wall_mask == 255)
    z_all = depth_map[vy, vx]
    valid = z_all > 0
    vy, vx, z_all = vy[valid], vx[valid], z_all[valid]

    x = (vx - cx) * z_all / fx
    y = (vy - cy) * z_all / fy
    all_points = np.vstack([x, y, z_all]).T  # (N, 3)

    # Adaptive distance threshold — mirrors extract_room_corners
    valid_z = depth_map[depth_map > 0]
    median_z = float(np.median(valid_z))
    dist_thresh = median_z * 0.05
    normal_radius = median_z * 0.05

    # Track which original indices are still available
    remaining = np.arange(len(all_points))
    accepted_normals = []
    wall_groups = []

    for _ in range(MAX_WALLS):
        if len(remaining) < 1000:
            break

        sub_pts = all_points[remaining]
        sub_pcd = o3d.geometry.PointCloud()
        sub_pcd.points = o3d.utility.Vector3dVector(sub_pts)
        sub_pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius, max_nn=30
            )
        )

        plane_model, sub_inliers = find_plane_ransac(sub_pcd, distance_threshold=dist_thresh)

        n_new = np.array(plane_model[:3])
        if any(np.dot(n_new, n_prev) > DUP_THRESHOLD for n_prev in accepted_normals):
            # Discard duplicate / parallel wall; still remove inliers to avoid loops
            remaining = np.delete(remaining, sub_inliers)
            continue

        accepted_normals.append(n_new)

        orig_inliers = remaining[sub_inliers]
        wall_groups.append((plane_model, vy[orig_inliers], vx[orig_inliers]))
        remaining = np.delete(remaining, sub_inliers)

    return wall_groups


def draw_wall_overlays(frame: np.ndarray, wall_groups, alpha: float) -> np.ndarray:
    """Blend a distinct semi-transparent colour over each wall's pixels."""
    out = frame.copy()
    for i, (_, row_idx, col_idx) in enumerate(wall_groups):
        color = WALL_COLORS[i % len(WALL_COLORS)]
        overlay = out.copy()
        overlay[row_idx, col_idx] = color
        out = cv2.addWeighted(out, 1.0 - alpha, overlay, alpha, 0)
    return out


def draw_corners(img: np.ndarray, corners_3d, K: np.ndarray) -> np.ndarray:
    """Project 3D corners to 2D and draw labelled circles."""
    out = img.copy()
    h, w = out.shape[:2]
    for i, corner in enumerate(corners_3d):
        u, v = project_3d_to_2d(corner, K)
        if not (0 <= u < w and 0 <= v < h):
            continue
        cv2.circle(out, (u, v), CORNER_RADIUS, CORNER_COLOR, -1, cv2.LINE_AA)
        cv2.circle(out, (u, v), CORNER_RADIUS, CORNER_BORDER, 2,  cv2.LINE_AA)
        label = f"C{i + 1}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.putText(out, label, (u - tw // 2, v + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CORNER_BORDER, 3, cv2.LINE_AA)
        cv2.putText(out, label, (u - tw // 2, v + th // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CORNER_COLOR, 1, cv2.LINE_AA)
    return out


def main():
    print(f"Loading frame 0 from {VIDEO_PATH}")
    cap = cv2.VideoCapture(VIDEO_PATH)
    ret, frame = cap.read()
    cap.release()
    assert ret, "Could not read first frame"

    print(f"Loading depth: {DEPTH_PATH}")
    depth = np.load(DEPTH_PATH)

    print(f"Loading masks: {MASK_NPZ}")
    npz = np.load(MASK_NPZ)
    floor_mask = npz["floor"]
    wall_mask  = npz["wall"]

    print(f"Loading K from {CALIB_JSON}")
    K = load_K(CALIB_JSON, CAMERA_ID)

    print("Extracting wall planes with pixel tracking...")
    wall_groups = extract_walls_with_pixels(depth, wall_mask, K)
    print(f"  Found {len(wall_groups)} wall planes")

    print("Extracting 3D corners...")
    corners_3d = extract_room_corners(depth, floor_mask, wall_mask, K)
    print(f"  Found {len(corners_3d)} corners")
    for i, c in enumerate(corners_3d):
        uv = project_3d_to_2d(c, K)
        print(f"    corner {i}: 3D={c.round(3)}  2D={uv}")

    print("Composing visualization...")
    vis = draw_wall_overlays(frame, wall_groups, WALL_ALPHA)
    vis = draw_corners(vis, corners_3d, K)

    out_path = os.path.join(OUTPUT_DIR, "corners_overlay.png")
    cv2.imwrite(out_path, vis)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
