"""
Ray-matching acceptance cone visualization.

For each annotated corner in a camera image, draws the angular acceptance cone:
the region in image space within which a detected corner direction would be
accepted as a match to that annotation.

Geometry
--------
  Matching compares world-space directions, but since R is orthogonal the
  angular distance is preserved in camera space. So the cone can be drawn
  entirely in camera space:

  1. Backproject annotation pixel:  d = normalize(K⁻¹ @ [u, v, 1])
  2. Build orthonormal frame around d:  e1 = normalize(arbitrary × d), e2 = d × e1
  3. Sample N boundary rays:  d_edge(θ) = cos(α)*d + sin(α)*(cos(θ)*e1 + sin(θ)*e2)
     where α = MAX_ANGLE_DEG (half-angle of the cone)
  4. Project each d_edge through K → image polygon

Output: plots/ray_matching_vis.png
"""

import os
import sys
import json
import math
import numpy as np
import cv2

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CAMERA_ID       = "Camera_06"
BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PATH      = os.path.join(BASE_DIR, "dataset/MTMC_Tracking_2025/val/Hospital_000/images", f"{CAMERA_ID}.jpg")
CALIB_JSON      = os.path.join(BASE_DIR, "dataset/MTMC_Tracking_2025/val/Hospital_000/calibration.json")
CAM_ANNS_JSON   = os.path.join(BASE_DIR, "dataset/MTMC_Tracking_2025/annotations/val/Hospital_000/annotated_cameras.json")
OUTPUT_PATH     = os.path.join(BASE_DIR, "plots/ray_matching_vis.png")

MAX_ANGLE_DEG   = 4.5        # acceptance half-angle — matches the pipeline threshold
CONE_SAMPLES    = 360        # smoothness of the cone polygon
CONE_ALPHA      = 0.30       # fill opacity
DOT_RADIUS      = 10
LABEL_OFFSET    = (14, -10)

# Distinct BGR colours per annotation point
PALETTE = [
    (  0, 200, 255),   # yellow
    ( 50, 200,  50),   # green
    (255, 100,  50),   # blue
    (  0, 100, 255),   # orange-red
    (200,  50, 200),   # purple
    (  0, 220, 180),   # lime
]
# ---------------------------------------------------------------------------


def load_K(calib_path, camera_id):
    with open(calib_path) as f:
        data = json.load(f)
    sensors = [s for s in data["sensors"] if s["type"] == "camera"]
    cam = next(s for s in sensors if s["id"] == camera_id)
    return np.array(cam["intrinsicMatrix"], dtype=np.float64)


def _cone_polygon(u, v, K, half_angle_deg, n_samples=360):
    """
    Return the image-space polygon (Nx2 int32) for the acceptance cone
    centred on annotation pixel (u, v) with given half-angle.

    Steps:
      1. Backproject (u, v) → unit camera-space direction d.
      2. Build orthonormal complement frame {e1, e2} ⊥ d.
      3. Rotate d by half_angle toward each azimuth in [0, 2π].
      4. Project each rotated ray back through K.
    """
    K_inv = np.linalg.inv(K)
    half  = math.radians(half_angle_deg)

    # Center ray in camera space
    d = K_inv @ np.array([u, v, 1.0], dtype=np.float64)
    d /= np.linalg.norm(d)

    # Orthonormal frame: pick an axis not parallel to d
    ref = np.array([0.0, 0.0, 1.0]) if abs(d[2]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = np.cross(ref, d)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(d, e1)   # already unit length

    # Sample cone boundary
    thetas = np.linspace(0, 2 * math.pi, n_samples, endpoint=False)
    pts = []
    for theta in thetas:
        d_edge = (math.cos(half) * d
                  + math.sin(half) * (math.cos(theta) * e1 + math.sin(theta) * e2))
        # Project: d_edge is a direction (depth 1 along Z is fine for projection)
        p = K @ d_edge
        if abs(p[2]) < 1e-8:
            continue
        px, py = p[0] / p[2], p[1] / p[2]
        pts.append((px, py))

    return np.array(pts, dtype=np.float32).reshape(-1, 1, 2)


def draw_cone(canvas, overlay, u, v, K, half_angle_deg, color, n_samples=360):
    """
    Draw the acceptance cone for one annotation onto `canvas` (in-place).
    `overlay` is a pre-allocated same-size image for alpha blending.
    """
    poly = _cone_polygon(u, v, K, half_angle_deg, n_samples)
    poly_i = poly.astype(np.int32)

    # Filled semi-transparent interior
    cv2.fillPoly(overlay, [poly_i], color)

    # Solid border
    cv2.polylines(canvas, [poly_i], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

    # Spokes: lines from center to cone edge at 0°, 90°, 180°, 270°
    n = len(poly)
    for idx in [0, n // 4, n // 2, 3 * n // 4]:
        ex, ey = int(poly[idx, 0, 0]), int(poly[idx, 0, 1])
        cv2.line(canvas, (int(u), int(v)), (ex, ey), color, 1, cv2.LINE_AA)


def main():
    print(f"Loading frame: {IMAGE_PATH}")
    frame = cv2.imread(IMAGE_PATH)
    assert frame is not None, f"Could not read {IMAGE_PATH}"
    H, W = frame.shape[:2]

    K = load_K(CALIB_JSON, CAMERA_ID)
    print(f"K loaded — fx={K[0,0]:.1f} fy={K[1,1]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    with open(CAM_ANNS_JSON) as f:
        all_cam_anns = json.load(f)

    anns = all_cam_anns.get(CAMERA_ID, [])
    if not anns:
        raise RuntimeError(f"No annotations found for {CAMERA_ID} in {CAM_ANNS_JSON}")
    print(f"Annotations: {[a['point_id'] for a in anns]}")

    # Build composite: canvas receives solid elements; overlay receives filled cones
    canvas  = frame.copy()
    overlay = frame.copy()

    for idx, ann in enumerate(anns):
        color   = PALETTE[idx % len(PALETTE)]
        pid     = ann["point_id"]
        u, v    = float(ann["x"]), float(ann["y"])

        print(f"  {pid}  ({u:.1f}, {v:.1f})")

        # Cone polygon (filled into overlay, border into canvas)
        draw_cone(canvas, overlay, u, v, K, MAX_ANGLE_DEG, color, CONE_SAMPLES)

        # Center dot
        cv2.circle(canvas,  (int(u), int(v)), DOT_RADIUS, color, -1, cv2.LINE_AA)
        cv2.circle(canvas,  (int(u), int(v)), DOT_RADIUS, (0, 0, 0), 2,  cv2.LINE_AA)

        # Point ID label
        lx = int(u) + LABEL_OFFSET[0]
        ly = int(v) + LABEL_OFFSET[1]
        cv2.putText(canvas, pid, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, pid, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color,    2, cv2.LINE_AA)

    # Alpha-blend filled cone overlay
    result = cv2.addWeighted(canvas, 1.0 - CONE_ALPHA, overlay, CONE_ALPHA, 0)

    # Redraw solid elements (dots, borders, labels) on top of blended image
    for idx, ann in enumerate(anns):
        color = PALETTE[idx % len(PALETTE)]
        pid   = ann["point_id"]
        u, v  = float(ann["x"]), float(ann["y"])

        poly  = _cone_polygon(u, v, K, MAX_ANGLE_DEG, CONE_SAMPLES)
        poly_i = poly.astype(np.int32)
        cv2.polylines(result, [poly_i], isClosed=True, color=color, thickness=2, lineType=cv2.LINE_AA)

        n = len(poly)
        for spoke_idx in [0, n // 4, n // 2, 3 * n // 4]:
            ex, ey = int(poly[spoke_idx, 0, 0]), int(poly[spoke_idx, 0, 1])
            cv2.line(result, (int(u), int(v)), (ex, ey), color, 1, cv2.LINE_AA)

        cv2.circle(result, (int(u), int(v)), DOT_RADIUS, color,   -1, cv2.LINE_AA)
        cv2.circle(result, (int(u), int(v)), DOT_RADIUS, (0,0,0),  2, cv2.LINE_AA)

        lx = int(u) + LABEL_OFFSET[0]
        ly = int(v) + LABEL_OFFSET[1]
        cv2.putText(result, pid, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(result, pid, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color,    2, cv2.LINE_AA)

    # Legend
    legend_y = 35
    legend = f"Acceptance cone  alpha = {MAX_ANGLE_DEG} deg  ({CAMERA_ID})"
    cv2.putText(result, legend, (15, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0),       4, cv2.LINE_AA)
    cv2.putText(result, legend, (15, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imwrite(OUTPUT_PATH, result)
    print(f"\nSaved → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
