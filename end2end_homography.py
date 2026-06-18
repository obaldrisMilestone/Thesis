"""
End-to-End Homography Estimation Pipeline
==========================================
Loads pre-computed depth maps and segmentation masks for a single camera,
then runs the full pipeline:

  Stage 1 — Floor geometry   : floor normal, camera height, pitch & roll
  Stage 2 — Corner detection : 3-D room corners via depth + RANSAC
  Stage 3 — Corner matching  : pair detected corners with annotated floorplan corners
  Stage 4 — Homography       : BEV warp (from normal) + point-based warp (from matches)

All heavy lifting is delegated to the existing modules in geometry/ and
cameras_and_floorplans/ — this script only orchestrates them.
"""

import os
import sys
import json
import math
import numpy as np
import cv2

# ── import sibling modules ─────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, 'geometry'))
sys.path.insert(0, os.path.join(_ROOT, 'cameras_and_floorplans'))

from normals import MetricFloorNormalExtractor
from corners_extraction import extract_room_corners, project_3d_to_2d
from corner_matcher import (
    direction_attr_to_world_azimuth,
    build_rotation_matrix,
    match_corners,
    build_homography_pairs,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  — edit this section to point at a different scenario / camera
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR = '/home/user/thesis/code/dataset/MTMC_Tracking_2025'
SCENARIO = 'val/Hospital_000'
SCENARIO_DIR = os.path.join(BASE_DIR, SCENARIO)
CAMERA_ID    = 'Camera_08'

# Paths to pre-computed artefacts (None = auto-detect from SCENARIO_DIR)
DEPTH_PATH   = os.path.join(SCENARIO_DIR, 'depth_maps_normalized', f'{CAMERA_ID}_bg.npy')  # .npy  — uint16 mm (GT) or float32 relative (DepthAnything)
MASK_PATH    = os.path.join('/home/user/thesis/code/segmentation/temporal_masks2', f'{CAMERA_ID}_temporal_bg.jpg')   # .npz  — keys 'floor','wall'  OR  colour-coded .jpg
ANNOTATIONS_DIR = os.path.join(BASE_DIR, 'annotations', SCENARIO)  # directory containing annotated_floorplan.json + annotated_cameras.json

# Corner-matching parameters
MAX_MATCH_ANGLE_DEG = 4.5   # reject a match if angular error exceeds this

# BEV canvas
BEV_PX_PER_M = 50
BEV_SIZE      = (800, 800)   # (width, height) in pixels

# Output
OUTPUT_DIR = None   # None → <SCENARIO_DIR>/end2end_out/<CAMERA_ID>

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def load_calibration(scenario_dir, camera_id):
    """
    Parse calibration.json for one camera.

    Independent fields (not from extrinsic — safe to use in estimation):
        K            : (3,3) intrinsic matrix
        cam_xy       : (2,) camera world XY from coordinates.x/y  ← NOT from R,t
        yaw_deg      : camera azimuth from the 'direction' attribute  ← NOT from R,t
        scale        : scaleFactor (floorplan pixels per metre)
        trans_x/y    : translationToGlobalCoordinates

    GT-only fields (used solely for error reporting, never fed into estimation):
        height_gt       : ground-truth camera height (metres)
        floor_normal_gt : (3,) GT floor normal in camera space
        pitch_gt/roll_gt: GT pitch and roll in degrees
    """
    path = os.path.join(scenario_dir, 'calibration.json')
    with open(path) as f:
        data = json.load(f)

    sensors = [s for s in data['sensors'] if s['type'] == 'camera']
    cam = next((s for s in sensors if s['id'] == camera_id), None)
    if cam is None:
        raise ValueError(f"Camera '{camera_id}' not found in {path}")

    K = np.array(cam['intrinsicMatrix'], dtype=np.float64)

    # ── independent of extrinsic ──────────────────────────────────────────────
    cam_xy = np.array([cam['coordinates']['x'], cam['coordinates']['y']])

    yaw_deg = 0.0
    for attr in cam.get('attributes', []):
        if attr['name'] == 'direction':
            yaw_deg = float(attr['value'])

    # ── GT only — for comparison/reporting, never used in estimation ──────────
    ext  = np.array(cam['extrinsicMatrix'], dtype=np.float64)
    R_gt = ext[:, :3]
    t_gt = ext[:, 3]
    cam_world_gt = -R_gt.T @ t_gt
    height_gt    = abs(float(cam_world_gt[2]))

    fn_gt = R_gt[:, 2].copy()
    if fn_gt[1] > 0:
        fn_gt = -fn_gt
    pitch_gt, roll_gt = _pitch_roll(fn_gt)

    return {
        'K':               K,
        'cam_xy':          cam_xy,
        'yaw_deg':         yaw_deg,
        'scale':           float(cam['scaleFactor']),
        'trans_x':         float(cam['translationToGlobalCoordinates']['x']),
        'trans_y':         float(cam['translationToGlobalCoordinates']['y']),
        # GT for error reporting only:
        'height_gt':       height_gt,
        'floor_normal_gt': fn_gt,
        'pitch_gt':        pitch_gt,
        'roll_gt':         roll_gt,
        'R_gt':            R_gt,
        'cam_world_gt':    cam_world_gt,
    }


def load_depth(depth_path):
    """
    Load a pre-computed depth map as float32 metres.

    Supports:
    - uint16 .npy  : dataset GT depth in millimetres (divides by 1000)
    - float32 .npy : relative depth from DepthAnything (returned as-is;
                     normals will still be correct, but height is non-metric)
    """
    arr = np.load(depth_path)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 1000.0
    return arr.astype(np.float32)


def load_masks(mask_path):
    """
    Load floor and wall binary masks (uint8, 255 = class).

    Supports:
    - .npz  with keys 'floor' and 'wall'
    - colour-coded .jpg/.png  (green=floor, red=wall per the pipeline convention)

    Returns (floor_mask, wall_mask) or (None, None) if the file does not exist.
    """
    if mask_path is None or not os.path.exists(mask_path):
        return None, None

    if mask_path.endswith('.npz'):
        data = np.load(mask_path)
        floor_mask = data.get('floor', None)
        wall_mask  = data.get('wall', None)
        return floor_mask, wall_mask

    # Colour-coded image: HSV thresholding
    img = cv2.imread(mask_path)
    if img is None:
        return None, None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Floor = green  BGR [0,255,0]
    floor_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([80, 255, 255]))

    # Wall = red (wraps at hue=0/180)
    wall_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,  50, 50]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255])),
    )
    return floor_mask, wall_mask


def load_frame(scenario_dir, camera_id):
    """Load the camera RGB frame (for visualisation)."""
    path = os.path.join(scenario_dir, 'images', f'{camera_id}.jpg')
    if not os.path.exists(path):
        return None
    return cv2.imread(path)


def load_floorplan(scenario_dir):
    """Load the floorplan map image."""
    path = os.path.join(scenario_dir, 'map.png')
    return cv2.imread(path)


def load_annotations(annotations_dir):
    """
    Load annotated_floorplan.json and annotated_cameras.json.

    Returns (floorplan_anns, cameras_anns) where:
        floorplan_anns : {point_id: {x, y}}   — floorplan pixel coordinates
        cameras_anns   : {camera_id: [{point_id, x, y}]}  — image pixel coordinates
    """
    fp_path  = os.path.join(annotations_dir, 'annotated_floorplan.json')
    cam_path = os.path.join(annotations_dir, 'annotated_cameras.json')

    fp_anns  = json.load(open(fp_path))  if os.path.exists(fp_path)  else {}
    cam_anns = json.load(open(cam_path)) if os.path.exists(cam_path) else {}
    return fp_anns, cam_anns


def _resolve_paths(scenario_dir, camera_id, depth_path, mask_path, annotations_dir, output_dir):
    """Fill in None paths with auto-detected defaults."""
    depth_maps_dir = os.path.join(scenario_dir, 'depth_maps')

    if depth_path is None:
        depth_path = os.path.join(depth_maps_dir, f'{camera_id}_bg.npy')

    if mask_path is None:
        # Try NPZ first, then colour JPG
        for candidate in [
            os.path.join(depth_maps_dir,  f'{camera_id}_masks.npz'),
            os.path.join(depth_maps_dir,  f'{camera_id}_bg.npz'),
            os.path.join(scenario_dir, 'segmentation', f'{camera_id}_masks.npz'),
        ]:
            if os.path.exists(candidate):
                mask_path = candidate
                break

    if annotations_dir is None:
        _marker = 'MTMC_Tracking_2025'
        parts = scenario_dir.split(_marker)
        scenario_name = parts[-1].strip('/')
        annotations_dir = os.path.join(_ROOT, 'cameras_and_floorplans', 'annotations', scenario_name)

    if output_dir is None:
        output_dir = os.path.join(scenario_dir, 'end2end_out', camera_id)

    return depth_path, mask_path, annotations_dir, output_dir


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _pitch_roll(floor_normal):
    """
    Pitch and roll (degrees) from a floor normal in camera space.

    The normal must point 'up' (Y < 0 in OpenCV convention).
    - Pitch: tilt around camera X axis — positive means camera looks down
    - Roll : tilt around camera Z axis — positive means right side down

    When the camera is perfectly horizontal, floor_normal = [0, -1, 0]
    and both pitch and roll are 0.
    """
    n = np.array(floor_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    if n[1] > 0:
        n = -n
    pitch = math.degrees(math.atan2(-n[2], -n[1]))
    roll  = math.degrees(math.atan2( n[0], -n[1]))
    return pitch, roll


# ── coordinate conversion (scale used exactly once, here, not inside matcher) ─

def fp_anns_to_world(fp_anns, map_h, scale, trans_x, trans_y):
    """
    Convert annotated floorplan pixels → world 3D (Z=0, floor plane).

    This is the single point where scale and translation are applied.
    The result is passed directly to match_corners; the matcher never sees scale.

    Formula (inverse of the annotator's world→pixel mapping):
        world_x = fp_x / scale  - trans_x
        world_y = (map_h - fp_y) / scale - trans_y
    """
    return {
        pid: np.array([
            coords['x'] / scale - trans_x,
            (map_h - coords['y']) / scale - trans_y,
            0.0,
        ])
        for pid, coords in fp_anns.items()
    }


def build_estimated_pose(calib, floor_result):
    """
    Build (R_est, cam_world_est) from independently known/estimated parameters.

    R_est        — world-to-camera rotation built from:
                     • yaw   : 'direction' attribute (independent of extrinsic)
                     • pitch : estimated by Stage 1 floor normal (or 0 if failed)
                     • roll  : estimated by Stage 1 floor normal (or 0 if failed)
    cam_world_est — [cam_xy[0], cam_xy[1], estimated_height]:
                     • XY    : calibration 'coordinates' field (independent of extrinsic)
                     • Z     : estimated height from Stage 1 (or 0 if failed)

    Neither R_est nor cam_world_est touches the GT extrinsic matrix.
    """
    world_az = direction_attr_to_world_azimuth(calib['yaw_deg'])

    pitch = floor_result['pred_pitch'] if floor_result else 0.0
    roll  = floor_result['pred_roll']  if floor_result else 0.0
    h_est = floor_result['camera_height'] if floor_result else 0.0

    R_est       = build_rotation_matrix(world_az, pitch, roll)
    cam_world_est = np.array([calib['cam_xy'][0], calib['cam_xy'][1], h_est])

    return R_est, cam_world_est


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STAGES
# ══════════════════════════════════════════════════════════════════════════════

def stage_floor_geometry(depth, floor_mask, K, calib):
    """
    Stage 1 — Floor normal, camera height, pitch & roll.

    Uses MetricFloorNormalExtractor (normals.py): masks depth to floor pixels,
    unprojects to 3D with K, fits a plane with RANSAC.

    Returns dict with predicted and GT values for comparison.
    """
    print("\n=== Stage 1: Floor Geometry ===")

    if floor_mask is None:
        print("  [!] No floor mask — skipping floor geometry estimation.")
        return None

    extractor = MetricFloorNormalExtractor()
    pred_normal, pred_height = extractor.extract_normal(depth, floor_mask, K)

    if pred_normal is None:
        print("  [!] RANSAC failed to fit a floor plane.")
        return None

    pred_pitch, pred_roll = _pitch_roll(pred_normal)
    gt_pitch,   gt_roll   = calib['pitch_gt'], calib['roll_gt']

    # Angular error between predicted and GT floor normals
    cos_a = float(np.clip(np.dot(pred_normal / np.linalg.norm(pred_normal),
                                  calib['floor_normal_gt']), -1.0, 1.0))
    ang_err = math.degrees(math.acos(cos_a))

    print(f"  Predicted floor normal : [{pred_normal[0]:+.4f}, {pred_normal[1]:+.4f}, {pred_normal[2]:+.4f}]")
    print(f"  GT floor normal        : [{calib['floor_normal_gt'][0]:+.4f}, {calib['floor_normal_gt'][1]:+.4f}, {calib['floor_normal_gt'][2]:+.4f}]")
    print(f"  Angular error          : {ang_err:.2f}°")
    print()
    print(f"  {'':12s}  {'Predicted':>10s}  {'GT':>10s}  {'Error':>8s}")
    print(f"  {'Pitch (°)':12s}  {pred_pitch:>10.2f}  {gt_pitch:>10.2f}  {abs(pred_pitch-gt_pitch):>8.2f}")
    print(f"  {'Roll  (°)':12s}  {pred_roll:>10.2f}  {gt_roll:>10.2f}  {abs(pred_roll-gt_roll):>8.2f}")


    return {
        'floor_normal': pred_normal,
        'camera_height': pred_height,
        'pred_pitch': pred_pitch,
        'pred_roll':  pred_roll,
        'angular_error_deg': ang_err,
    }


def stage_corner_detection(depth, floor_mask, wall_mask, K):
    """
    Stage 2 — 3D room corner detection.

    Delegates entirely to extract_room_corners (corners_extraction.py):
    unprojects depth to 3D, fits floor + up to 4 wall planes with RANSAC,
    intersects floor ∩ wall_i ∩ wall_j triples.

    Returns list of np.ndarray (3,) corner positions in camera space, or []
    if masks are unavailable.
    """
    print("\n=== Stage 2: Corner Detection ===")

    if floor_mask is None or wall_mask is None:
        print("  [!] Floor or wall mask missing — skipping corner detection.")
        return []

    corners = extract_room_corners(depth, floor_mask, wall_mask, K)
    print(f"  Found {len(corners)} candidate 3D corner(s).")
    for i, c in enumerate(corners):
        print(f"    Corner {i}: cam-space [{c[0]:+.2f}, {c[1]:+.2f}, {c[2]:+.2f}]")

    return corners


def stage_corner_matching(corners_cam, calib, floor_result, fp_anns, map_h):
    """
    Stage 3 — Match detected corners to annotated floorplan corners.

    Uses only estimated/independent parameters — no GT extrinsic:
      • R_est       : built from direction attribute (yaw) + Stage 1 pitch/roll
      • cam_world   : calibration coordinates XY + Stage 1 estimated height
      • ann_world   : floorplan annotations converted to world 3D (scale used once here)

    Delegates to match_corners (corner_matcher.py).
    """
    print("\n=== Stage 3: Corner Matching ===")

    if not corners_cam or not fp_anns:
        reason = "no detected corners" if not corners_cam else "no floorplan annotations"
        print(f"  [!] Skipping — {reason}.")
        return [], {}

    # Build estimated pose from known yaw + Stage 1 pitch/roll/height
    R_est, cam_world_est = build_estimated_pose(calib, floor_result)

    print(f"  Estimated pose:")
    print(f"    cam_world = [{cam_world_est[0]:.2f}, {cam_world_est[1]:.2f}, {cam_world_est[2]:.2f}]  "
          f"(XY from coordinates, Z from Stage 1)")
    print(f"    yaw={calib['yaw_deg']:.1f}°  "
          f"pitch={floor_result['pred_pitch']:.2f}°  "
          f"roll={floor_result['pred_roll']:.2f}°" if floor_result
          else f"    yaw={calib['yaw_deg']:.1f}°  pitch=0°  roll=0° (Stage 1 unavailable)")

    # Convert annotated floorplan pixels → world 3D (the only use of scale)
    ann_world = fp_anns_to_world(fp_anns, map_h,
                                 calib['scale'], calib['trans_x'], calib['trans_y'])

    matches = match_corners(
        corners_cam,
        R_est,
        cam_world_est,
        ann_world,
        max_angle_deg=MAX_MATCH_ANGLE_DEG,
    )

    n_matched = sum(1 for m in matches if m['point_id'] is not None)
    print(f"  Matched {n_matched}/{len(matches)} corner(s) "
          f"(threshold: {MAX_MATCH_ANGLE_DEG}°)")

    for m in matches:
        pid  = m['point_id'] or "(no match)"
        err  = m['angular_err_deg']
        flag = "" if m['point_id'] else f"  [best was {err:.1f}° > {MAX_MATCH_ANGLE_DEG}°]"
        print(f"    Corner {m['corner_idx']} → {pid:8s}  angular err {err:.1f}°{flag}")

    return matches, ann_world


def _estimate_height_from_best_match(matches, corners_cam, ann_world, calib, R_est):
    """
    Recover metric camera height from the single best-matched corner.

    The depth map is relative (DepthAnything), so the corner position in camera
    space is in arbitrary units.  We know the horizontal ground-plane distance
    from the camera to the matched annotation (in metres), so we can compute the
    scale factor and from it the metric height.

    Derivation:
        The world-space offset from camera to corner is  s * R^T @ corner_cam,
        where s is the unknown scale.  The horizontal magnitude of that offset
        must equal d_horiz (known from the floorplan annotation), giving:

            s = d_horiz / || (R^T @ corner_cam)[:2] ||

        Height is then  h = -s * (R^T @ corner_cam)[2]   (positive, Z points up).

    Returns (h_est, scale, best_match) or (None, None, None) on failure.
    """
    valid = [m for m in matches if m['point_id'] is not None]
    if not valid:
        return None, None, None

    best = min(valid, key=lambda m: m['angular_err_deg'])
    corner_cam = np.array(corners_cam[best['corner_idx']], dtype=np.float64)

    P_ann = ann_world[best['point_id']]          # [Wx, Wy, 0] in world metres
    cx, cy = float(calib['cam_xy'][0]), float(calib['cam_xy'][1])
    d_horiz = math.sqrt((P_ann[0] - cx) ** 2 + (P_ann[1] - cy) ** 2)
    if d_horiz < 1e-6:
        return None, None, None

    v_world     = R_est.T @ corner_cam           # corner direction in world (relative)
    d_rel_horiz = math.sqrt(v_world[0] ** 2 + v_world[1] ** 2)
    if d_rel_horiz < 1e-8:
        return None, None, None

    s     = d_horiz / d_rel_horiz
    h_est = float(-s * v_world[2])               # v_world[2] < 0 for floor corner

    if h_est <= 0:
        return None, None, None

    return h_est, float(s), best


def _build_floor_homography(K, R, cam_world, calib, map_h):
    """
    Analytical homography  H : camera image pixel → floorplan pixel.

    Covers only the floor plane (Z = 0 in world coordinates).

    H = M_world_to_fp @ inv( K @ [r1 | r2 | t] )

    where:
        r1, r2  = first two columns of R   (world X, Y axes in camera space)
        t       = -R @ cam_world
        M_world_to_fp converts world XY metres to floorplan pixel coordinates:
            fp_x =  scale * (world_x + trans_x)
            fp_y =  map_h  - scale * (world_y + trans_y)
    """
    cam_world = np.array(cam_world, dtype=np.float64)
    r1, r2    = R[:, 0], R[:, 1]
    t         = -R @ cam_world

    H_floor_to_img = K @ np.column_stack([r1, r2, t])   # 3×3

    scale, tx, ty = calib['scale'], calib['trans_x'], calib['trans_y']
    M_w2fp = np.array([
        [scale,   0.0,    scale * tx              ],
        [0.0,    -scale,  map_h - scale * ty       ],
        [0.0,    0.0,    1.0                       ],
    ])

    return M_w2fp @ np.linalg.inv(H_floor_to_img)


def _compare_homographies(H_est, H_gt, cam_image_pts, fp_anns, scale_px_per_m):
    """
    Report per-point reprojection errors using the annotated image↔floorplan pairs.

    Columns printed:
      GT err (px)   — distance between H_gt projection and annotated floorplan point
      Est err (px)  — distance between H_est projection and annotated floorplan point
      H_gt vs H_est — pixel distance between the two projections of the same image point
      metric err (m)— H_gt vs H_est converted to metres via scale_px_per_m
    """
    pairs = [(p, fp_anns[p['point_id']])
             for p in cam_image_pts if p['point_id'] in fp_anns]
    if not pairs:
        print("  No annotated pairs available for comparison.")
        return

    print(f"\n  {'Point':8s}  {'GT err (px)':>11s}  {'Est err (px)':>12s}"
          f"  {'GT↔Est (px)':>11s}  {'GT↔Est (m)':>10s}")
    errs_gt, errs_est, errs_metric = [], [], []

    for ann_cam, ann_fp in pairs:
        src = np.array([[[ann_cam['x'], ann_cam['y']]]], dtype=np.float32)
        dst = np.array([ann_fp['x'], ann_fp['y']], dtype=np.float64)
        pid = ann_cam['point_id']

        p_gt = p_est = None
        e_gt = e_est = e_diff = e_metric = float('nan')

        if H_gt is not None:
            p_gt  = cv2.perspectiveTransform(src, H_gt)[0, 0].astype(np.float64)
            e_gt  = float(np.linalg.norm(p_gt - dst))
            errs_gt.append(e_gt)

        if H_est is not None:
            p_est  = cv2.perspectiveTransform(src, H_est)[0, 0].astype(np.float64)
            e_est  = float(np.linalg.norm(p_est - dst))
            errs_est.append(e_est)

        if p_gt is not None and p_est is not None:
            e_diff   = float(np.linalg.norm(p_gt - p_est))
            e_metric = e_diff / scale_px_per_m
            errs_metric.append(e_metric)

        print(f"  {pid:8s}  {e_gt:>11.1f}  {e_est:>12.1f}"
              f"  {e_diff:>11.1f}  {e_metric:>10.3f}")

    if errs_gt or errs_est:
        mean_gt     = np.mean(errs_gt)     if errs_gt     else float('nan')
        mean_est    = np.mean(errs_est)    if errs_est    else float('nan')
        mean_metric = np.mean(errs_metric) if errs_metric else float('nan')
        mean_diff   = mean_metric * scale_px_per_m if errs_metric else float('nan')
        print(f"  {'Mean':8s}  {mean_gt:>11.1f}  {mean_est:>12.1f}"
              f"  {mean_diff:>11.1f}  {mean_metric:>10.3f}")


def stage_homography(corners_cam, matches, floor_result, calib,
                     ann_world, fp_anns, cam_image_pts, map_h):
    """
    Stage 4 — Analytical floor-to-floorplan homography.

    1. Build R_est from known yaw + Stage-1 pitch / roll.
    2. Recover metric camera height via the best-matched corner:
       the known horizontal floorplan distance to that point provides the
       depth scale factor.
    3. Build H_img_to_fp = M_world_to_fp @ inv(K [r1|r2|t])  analytically.
    4. Build H_gt the same way using GT extrinsic for comparison.
    5. Report per-point reprojection errors on the annotated pairs.
    """
    print("\n=== Stage 4: Homography ===")
    result = {'H_bev': None, 'H_fp': None, 'H_fp_gt': None, 'fp_anns': fp_anns}

    K = calib['K']

    # ── 1. R_est from known yaw + Stage-1 pitch / roll ───────────────────────
    world_az = direction_attr_to_world_azimuth(calib['yaw_deg'])
    pitch = floor_result['pred_pitch'] if floor_result else 0.0
    roll  = floor_result['pred_roll']  if floor_result else 0.0
    R_est = build_rotation_matrix(world_az, pitch, roll)

    # ── 2. Metric height from best matched corner ─────────────────────────────
    h_est, scale_factor, best_match = _estimate_height_from_best_match(
        matches, corners_cam, ann_world, calib, R_est
    )

    if h_est is None:
        print("  [!] Height estimation failed — no valid matched corner.")
        h_est = floor_result['camera_height'] if floor_result else 0.0
        print(f"      Using Stage 1 height: {h_est:.3f} (relative, not metric)")
    else:
        h_gt = calib['height_gt']
        print(f"  Best match    : corner {best_match['corner_idx']} "
              f"→ {best_match['point_id']}  "
              f"(angular err {best_match['angular_err_deg']:.2f}°)")
        print(f"  Depth scale   : {scale_factor:.4f}  (relative → metric)")
        print(f"  Height        : est {h_est:.3f} m   GT {h_gt:.3f} m   "
              f"err {abs(h_est - h_gt):.3f} m")

    # ── 3. Estimated homography ───────────────────────────────────────────────
    cam_world_est = np.array([calib['cam_xy'][0], calib['cam_xy'][1], h_est])
    H_fp = _build_floor_homography(K, R_est, cam_world_est, calib, map_h)
    result['H_fp'] = H_fp
    print(f"  Estimated H   : built from yaw={calib['yaw_deg']:.1f}°  "
          f"pitch={pitch:.2f}°  roll={roll:.2f}°  h={h_est:.3f} m")

    # ── 4. GT homography ──────────────────────────────────────────────────────
    H_fp_gt = _build_floor_homography(
        K, calib['R_gt'], calib['cam_world_gt'], calib, map_h
    )
    result['H_fp_gt'] = H_fp_gt
    print(f"  GT H          : built from GT extrinsic  "
          f"(h={calib['height_gt']:.3f} m)")

    # ── 5. Reprojection comparison on annotated pairs ─────────────────────────
    print("\n  Reprojection error  (image annotation → floorplan pixel):")
    _compare_homographies(H_fp, H_fp_gt, cam_image_pts, fp_anns, calib['scale'])

    return result


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

def _draw_corners_on_frame(frame, corners_cam, K, color=(0, 255, 255)):
    """Draw reprojected 3D corners on the camera frame."""
    out = frame.copy()
    h, w = out.shape[:2]
    for i, corner in enumerate(corners_cam):
        u, v = project_3d_to_2d(corner, K)
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(out, (u, v), 10, color, -1)
            cv2.circle(out, (u, v), 10, (0, 0, 0), 2)
            cv2.putText(out, str(i), (u + 14, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return out


def _draw_matches_on_frame(frame, corners_cam, matches, K):
    """Draw matched corners with their annotation label."""
    out = frame.copy()
    h, w = out.shape[:2]
    for m in matches:
        idx   = m['corner_idx']
        label = m['point_id'] or "?"
        color = (0, 200, 0) if m['point_id'] else (0, 0, 200)
        u, v  = project_3d_to_2d(corners_cam[idx], K)
        if 0 <= u < w and 0 <= v < h:
            cv2.circle(out, (u, v), 12, color, -1)
            cv2.circle(out, (u, v), 12, (0, 0, 0), 2)
            cv2.putText(out, label, (u + 14, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return out


def _draw_floorplan_matches(floorplan, matches, calib, fp_anns, map_h):
    """Draw matched corners on the floorplan using direct annotation pixel coords."""
    out = floorplan.copy()

    # Camera position in floorplan pixels (world→fp: fp_x = (wx+tx)*scale, fp_y = map_h-(wy+ty)*scale)
    cx = (calib['cam_xy'][0] + calib['trans_x']) * calib['scale']
    cy = map_h - (calib['cam_xy'][1] + calib['trans_y']) * calib['scale']
    cv2.drawMarker(out, (int(cx), int(cy)), (255, 0, 0),
                   cv2.MARKER_TRIANGLE_UP, 20, 3)

    for m in matches:
        if m['point_id'] is None:
            continue
        ann = fp_anns[m['point_id']]
        fx, fy = int(ann['x']), int(ann['y'])
        cv2.circle(out, (fx, fy), 10, (0, 200, 0), -1)
        cv2.circle(out, (fx, fy), 10, (0, 0, 0), 2)
        cv2.putText(out, m['point_id'], (fx + 12, fy - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    return out


def _draw_stage3_matching(frame, corners_cam, matches, cam_anns, camera_id, K):
    """
    After Stage 3: overlay annotated points (blue), predicted corners (yellow),
    and lines connecting each matched pair on the original frame.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Annotated camera-image points for this camera
    ann_pts = {p['point_id']: (int(p['x']), int(p['y']))
               for p in cam_anns.get(camera_id, [])}

    # Project all detected corners to 2D
    proj = {}
    for i, corner in enumerate(corners_cam):
        u, v = project_3d_to_2d(corner, K)
        if 0 <= u < w and 0 <= v < h:
            proj[i] = (u, v)

    # Draw lines first (underneath the dots)
    for m in matches:
        if m['point_id'] is None:
            continue
        pid = m['point_id']
        ci  = m['corner_idx']
        if pid in ann_pts and ci in proj:
            cv2.line(out, proj[ci], ann_pts[pid], (200, 200, 200), 2, cv2.LINE_AA)

    # Draw predicted corners (yellow)
    for ci, (u, v) in proj.items():
        cv2.circle(out, (u, v), 10, (0, 215, 255), -1)
        cv2.circle(out, (u, v), 10, (0, 0, 0), 2)
        cv2.putText(out, str(ci), (u + 13, v - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

    # Draw annotated points (blue)
    for pid, (ax, ay) in ann_pts.items():
        cv2.circle(out, (ax, ay), 10, (255, 80, 0), -1)
        cv2.circle(out, (ax, ay), 10, (0, 0, 0), 2)
        cv2.putText(out, pid, (ax + 13, ay - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 0), 2)

    # Legend
    lx, ly, ls = 14, h - 60, 0.55
    cv2.circle(out, (lx + 8, ly),      8, (0, 215, 255), -1)
    cv2.putText(out, "predicted corner", (lx + 22, ly + 5), cv2.FONT_HERSHEY_SIMPLEX, ls, (255, 255, 255), 1)
    cv2.circle(out, (lx + 8, ly + 26), 8, (255, 80, 0), -1)
    cv2.putText(out, "annotated point",  (lx + 22, ly + 31), cv2.FONT_HERSHEY_SIMPLEX, ls, (255, 255, 255), 1)

    return out


def save_outputs(output_dir, camera_id, frame, floorplan, corners_cam,
                 matches, calib, H_dict, map_h):
    """Save all visual outputs to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    def save(name, img):
        path = os.path.join(output_dir, f'{camera_id}_{name}')
        cv2.imwrite(path, img)
        print(f"  Saved: {path}")

    if frame is not None:
        if corners_cam:
            save('corners.jpg',
                 _draw_corners_on_frame(frame, corners_cam, calib['K']))
        if matches:
            save('matches.jpg',
                 _draw_matches_on_frame(frame, corners_cam, matches, calib['K']))

        if H_dict['H_bev'] is not None:
            bev = cv2.warpPerspective(frame, H_dict['H_bev'], BEV_SIZE)
            save('bev.jpg', bev)

        if floorplan is not None:
            fp_w, fp_h = floorplan.shape[1], floorplan.shape[0]
            alpha = 0.5
            if H_dict['H_fp'] is not None:
                warped = cv2.warpPerspective(frame, H_dict['H_fp'], (fp_w, fp_h))
                blend  = cv2.addWeighted(floorplan, 1 - alpha, warped, alpha, 0)
                save('fp_warp_blend.jpg', blend)
            if H_dict.get('H_fp_gt') is not None:
                warped_gt = cv2.warpPerspective(frame, H_dict['H_fp_gt'], (fp_w, fp_h))
                blend_gt  = cv2.addWeighted(floorplan, 1 - alpha, warped_gt, alpha, 0)
                save('fp_warp_blend_gt.jpg', blend_gt)

    if floorplan is not None and matches:
        save('floorplan_matches.jpg',
             _draw_floorplan_matches(floorplan, matches, calib, H_dict['fp_anns'], map_h))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(scenario_dir, camera_id,
        depth_path=None, mask_path=None, annotations_dir=None, output_dir=None):
    """
    Run the full end-to-end pipeline for one camera.

    All paths default to auto-detected locations relative to scenario_dir.
    Override individual paths by passing them explicitly.
    """

    # ── resolve paths ────────────────────────────────────────────────────────
    depth_path, mask_path, annotations_dir, output_dir = _resolve_paths(
        scenario_dir, camera_id, depth_path, mask_path, annotations_dir, output_dir
    )

    print(f"\n{'='*60}")
    print(f" End-to-End Homography  |  {camera_id}")
    print(f"{'='*60}")
    print(f" Scenario    : {scenario_dir}")
    print(f" Depth       : {depth_path}")
    print(f" Masks       : {mask_path or '(not found)'}")
    print(f" Annotations : {annotations_dir}")
    print(f" Output      : {output_dir}")

    # ── load data ────────────────────────────────────────────────────────────
    calib      = load_calibration(scenario_dir, camera_id)
    depth      = load_depth(depth_path)
    floor_mask, wall_mask = load_masks(mask_path)
    frame      = load_frame(scenario_dir, camera_id)
    floorplan  = load_floorplan(scenario_dir)
    fp_anns, cam_anns = load_annotations(annotations_dir)

    # Keep only floorplan points that were also annotated in this camera's image
    visible_ids = {p['point_id'] for p in cam_anns.get(camera_id, [])}
    fp_anns = {pid: v for pid, v in fp_anns.items() if pid in visible_ids}

    map_h = floorplan.shape[0] if floorplan is not None else 1080

    # ── pipeline ─────────────────────────────────────────────────────────────
    floor_result = stage_floor_geometry(depth, floor_mask, calib['K'], calib)

    corners_cam  = stage_corner_detection(depth, floor_mask, wall_mask, calib['K'])

    matches, ann_world = stage_corner_matching(
        corners_cam, calib, floor_result, fp_anns, map_h
    )

    if frame is not None:
        results_dir = os.path.join(_ROOT, 'results')
        os.makedirs(results_dir, exist_ok=True)
        stage3_img = _draw_stage3_matching(
            frame, corners_cam, matches, cam_anns, camera_id, calib['K']
        )
        stage3_path = os.path.join(results_dir, f'{camera_id}_stage3_matching.jpg')
        cv2.imwrite(stage3_path, stage3_img)
        print(f"\n  Stage 3 visualisation saved: {stage3_path}")

    H_dict = stage_homography(
        corners_cam, matches, floor_result, calib,
        ann_world, fp_anns,
        cam_anns.get(camera_id, []),
        map_h,
    )

    print(f"\nDone.")
    return {
        'calib':       calib,
        'floor':       floor_result,
        'corners_cam': corners_cam,
        'matches':     matches,
        'H_fp':        H_dict['H_fp'],
    }


if __name__ == "__main__":
    run(
        scenario_dir    = SCENARIO_DIR,
        camera_id       = CAMERA_ID,
        depth_path      = DEPTH_PATH,
        mask_path       = MASK_PATH,
        annotations_dir = ANNOTATIONS_DIR,
        output_dir      = OUTPUT_DIR,
    )
