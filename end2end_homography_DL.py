"""
End-to-End Homography Estimation Pipeline — DL Corner Detection
===============================================================
Drop-in replacement for end2end_homography.py that swaps Stage 2 (RANSAC
3D corner extraction) with a learned heatmap detector (KeypointDINO or
KeypointDINO_D depending on DL_MODEL below).

Stages:
  Stage 1 — Floor geometry   : floor normal, camera height, pitch & roll (unchanged)
  Stage 2 — DL corner detect : 2D keypoints from heatmap → backprojected camera rays
  Stage 3 — Corner matching  : pair detected corners with annotated floorplan corners (unchanged)
  Stage 4 — Homography       : analytical floor homography from matched corner (unchanged)

DL pipeline integration
───────────────────────
The DL model outputs 2D pixel keypoints.  Each is backprojected to a camera-
space ray via  ray = K⁻¹ @ [u, v, 1]ᵀ.  The rest of the pipeline treats these
rays as relative 3D corners — this is valid because:
  • match_corners uses only ray directions (scale-invariant)
  • _estimate_height_from_best_match works with relative depth (it computes
    the scale factor itself from the known horizontal floorplan distance)

The dino_d_rgb variant also consumes the depth map (same .npy loaded for
Stage 1) so no extra input is required.
"""

import importlib.util
import os
import sys
import json
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ── import sibling modules ─────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_ROOT, 'geometry'))
sys.path.insert(0, os.path.join(_ROOT, 'cameras_and_floorplans'))

from normals import MetricFloorNormalExtractor
from corners_extraction import project_3d_to_2d
from corner_matcher import (
    direction_attr_to_world_azimuth,
    build_rotation_matrix,
    match_corners,
)

# ══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION  — edit this section to point at a different scenario / camera
# ══════════════════════════════════════════════════════════════════════════════

BASE_DIR     = '/home/user/thesis/code/dataset/MTMC_Tracking_2025'
SCENARIO     = 'val/Hospital_000'
SCENARIO_DIR = os.path.join(BASE_DIR, SCENARIO)
CAMERA_ID    = 'Camera_08'

# Paths to pre-computed artefacts
DEPTH_PATH = os.path.join(SCENARIO_DIR, 'depth_maps_normalized', f'{CAMERA_ID}_bg.npy')
MASK_PATH  = os.path.join('/home/user/thesis/code/segmentation/temporal_masks2',
                          f'{CAMERA_ID}_temporal_bg.jpg')
ANNOTATIONS_DIR = os.path.join(BASE_DIR, 'annotations', SCENARIO)

# ── DL model ──────────────────────────────────────────────────────────────────
# 'dino_rgb'   — RGB-only  (KeypointDINO,   no depth input)
# 'dino_d_rgb' — RGB-D     (KeypointDINO_D, uses DEPTH_PATH at inference time)
DL_MODEL      = 'dino_d_rgb'
DL_CHECKPOINT = os.path.join(_ROOT, DL_MODEL, 'checkpoints', 'best_model.pth')
DL_THRESHOLD  = 0.3   # heatmap peak confidence threshold (0–1)

# Corner-matching parameters
MAX_MATCH_ANGLE_DEG = 4.5

# BEV canvas
BEV_PX_PER_M = 50
BEV_SIZE      = (800, 800)

# Output
OUTPUT_DIR = None   # None → <SCENARIO_DIR>/end2end_out_DL/<CAMERA_ID>

# ══════════════════════════════════════════════════════════════════════════════
# DL MODEL LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _load_training_module(model_variant: str):
    """
    Import dino_rgb/training.py or dino_d_rgb/training.py by file path so
    both can coexist without naming conflicts.
    """
    module_path = os.path.join(_ROOT, model_variant, 'training.py')
    if not os.path.exists(module_path):
        raise FileNotFoundError(f"Training module not found: {module_path}")
    spec = importlib.util.spec_from_file_location(f"training_{model_variant}", module_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_dl_model(checkpoint_path: str, model_variant: str, device: torch.device):
    """
    Reconstruct the DL model from a checkpoint.  Architecture parameters are
    read from the config stored inside the checkpoint.

    Returns (model, training_cfg, extract_keypoints_fn).
    """
    mod  = _load_training_module(model_variant)
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt['config']

    if model_variant == 'dino_rgb':
        model = mod.KeypointDINO(
            dino_channels=cfg['dino_channels'],
            neck_channels=cfg['neck_channels'],
            num_decoder_blocks=cfg['num_decoder_blocks'],
            num_classes=cfg['num_classes'],
            freeze_backbone=cfg.get('freeze_backbone', True),
        )
    else:  # dino_d_rgb
        model = mod.KeypointDINO_D(
            dino_channels=cfg['dino_channels'],
            depth_channels=cfg['depth_channels'],
            neck_channels=cfg['neck_channels'],
            num_decoder_blocks=cfg['num_decoder_blocks'],
            num_classes=cfg['num_classes'],
            freeze_rgb_backbone=cfg.get('freeze_rgb_backbone', True),
        )

    model.load_state_dict(ckpt['model_state_dict'])
    model.to(device).eval()

    epoch    = ckpt.get('epoch', '?')
    val_loss = ckpt.get('val_loss', float('nan'))
    print(f"  Loaded {model_variant} checkpoint  (epoch={epoch}, val_loss={val_loss:.4f})")

    return model, cfg, mod.extract_keypoints


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING  (identical to end2end_homography.py)
# ══════════════════════════════════════════════════════════════════════════════

def load_calibration(scenario_dir, camera_id):
    path = os.path.join(scenario_dir, 'calibration.json')
    with open(path) as f:
        data = json.load(f)
    sensors = [s for s in data['sensors'] if s['type'] == 'camera']
    cam = next((s for s in sensors if s['id'] == camera_id), None)
    if cam is None:
        raise ValueError(f"Camera '{camera_id}' not found in {path}")

    K      = np.array(cam['intrinsicMatrix'], dtype=np.float64)
    cam_xy = np.array([cam['coordinates']['x'], cam['coordinates']['y']])

    yaw_deg = 0.0
    for attr in cam.get('attributes', []):
        if attr['name'] == 'direction':
            yaw_deg = float(attr['value'])

    ext          = np.array(cam['extrinsicMatrix'], dtype=np.float64)
    R_gt         = ext[:, :3]
    t_gt         = ext[:, 3]
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
        'height_gt':       height_gt,
        'floor_normal_gt': fn_gt,
        'pitch_gt':        pitch_gt,
        'roll_gt':         roll_gt,
        'R_gt':            R_gt,
        'cam_world_gt':    cam_world_gt,
    }


def load_depth(depth_path):
    arr = np.load(depth_path)
    if arr.dtype == np.uint16:
        return arr.astype(np.float32) / 1000.0
    return arr.astype(np.float32)


def load_masks(mask_path):
    if mask_path is None or not os.path.exists(mask_path):
        return None, None
    if mask_path.endswith('.npz'):
        data = np.load(mask_path)
        return data.get('floor', None), data.get('wall', None)
    img = cv2.imread(mask_path)
    if img is None:
        return None, None
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    floor_mask = cv2.inRange(hsv, np.array([40,  50, 50]), np.array([80,  255, 255]))
    wall_mask  = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0,   50, 50]), np.array([10,  255, 255])),
        cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255])),
    )
    return floor_mask, wall_mask


def load_frame(scenario_dir, camera_id):
    path = os.path.join(scenario_dir, 'images', f'{camera_id}.jpg')
    return cv2.imread(path) if os.path.exists(path) else None


def load_floorplan(scenario_dir):
    return cv2.imread(os.path.join(scenario_dir, 'map.png'))


def load_annotations(annotations_dir):
    fp_path  = os.path.join(annotations_dir, 'annotated_floorplan.json')
    cam_path = os.path.join(annotations_dir, 'annotated_cameras.json')
    fp_anns  = json.load(open(fp_path))  if os.path.exists(fp_path)  else {}
    cam_anns = json.load(open(cam_path)) if os.path.exists(cam_path) else {}
    return fp_anns, cam_anns


def _resolve_paths(scenario_dir, camera_id, depth_path, mask_path, annotations_dir, output_dir):
    if depth_path is None:
        depth_path = os.path.join(scenario_dir, 'depth_maps', f'{camera_id}_bg.npy')
    if mask_path is None:
        for candidate in [
            os.path.join(scenario_dir, 'depth_maps',   f'{camera_id}_masks.npz'),
            os.path.join(scenario_dir, 'depth_maps',   f'{camera_id}_bg.npz'),
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
        output_dir = os.path.join(scenario_dir, 'end2end_out_DL', camera_id)
    return depth_path, mask_path, annotations_dir, output_dir


# ══════════════════════════════════════════════════════════════════════════════
# GEOMETRY HELPERS  (identical to end2end_homography.py)
# ══════════════════════════════════════════════════════════════════════════════

def _pitch_roll(floor_normal):
    n = np.array(floor_normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    if n[1] > 0:
        n = -n
    pitch = math.degrees(math.atan2(-n[2], -n[1]))
    roll  = math.degrees(math.atan2( n[0], -n[1]))
    return pitch, roll


def fp_anns_to_world(fp_anns, map_h, scale, trans_x, trans_y):
    return {
        pid: np.array([
            coords['x'] / scale - trans_x,
            (map_h - coords['y']) / scale - trans_y,
            0.0,
        ])
        for pid, coords in fp_anns.items()
    }


def build_estimated_pose(calib, floor_result):
    world_az = direction_attr_to_world_azimuth(calib['yaw_deg'])
    pitch = floor_result['pred_pitch'] if floor_result else 0.0
    roll  = floor_result['pred_roll']  if floor_result else 0.0
    h_est = floor_result['camera_height'] if floor_result else 0.0
    R_est         = build_rotation_matrix(world_az, pitch, roll)
    cam_world_est = np.array([calib['cam_xy'][0], calib['cam_xy'][1], h_est])
    return R_est, cam_world_est


# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE STAGES
# ══════════════════════════════════════════════════════════════════════════════

def stage_floor_geometry(depth, floor_mask, K, calib):
    """Stage 1 — unchanged from end2end_homography.py."""
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

    cos_a   = float(np.clip(np.dot(pred_normal / np.linalg.norm(pred_normal),
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
        'floor_normal':      pred_normal,
        'camera_height':     pred_height,
        'pred_pitch':        pred_pitch,
        'pred_roll':         pred_roll,
        'angular_error_deg': ang_err,
    }


def _preprocess_for_dl(frame_bgr, depth_np, image_size, device):
    """
    Prepare tensors for the DL model from the original frame and depth map.

    Returns (rgb_tensor [1,3,H,W], depth_tensor [1,1,H,W]) ready for inference.
    depth_tensor is None when running the RGB-only model.
    """
    img_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    rgb_tf = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    rgb_tensor = rgb_tf(image=img_rgb)['image'].unsqueeze(0).to(device)

    if depth_np is None:
        return rgb_tensor, None

    d = depth_np.astype(np.float32)
    d_min, d_max = d.min(), d.max()
    d = (d - d_min) / (d_max - d_min + 1e-8)
    d = cv2.resize(d, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    depth_tensor = torch.from_numpy(d).unsqueeze(0).unsqueeze(0).to(device)

    return rgb_tensor, depth_tensor


def _backproject_keypoints(keypoints_2d, orig_hw, model_image_size, K):
    """
    Convert DL-detected 2D pixel keypoints to camera-space rays.

    1. Scale from model input space (model_image_size × model_image_size)
       back to the original frame resolution.
    2. Backproject each pixel through K:  ray = K⁻¹ @ [u, v, 1]ᵀ

    These rays are scale-free 3D directions in camera space.  They are
    treated as 'corners_cam' by the rest of the pipeline, which only ever
    uses their direction (angle matching is scale-invariant, and height
    estimation computes its own scale factor from the floorplan distance).
    """
    orig_h, orig_w = orig_hw
    sx = orig_w / model_image_size
    sy = orig_h / model_image_size

    K_inv = np.linalg.inv(K)
    rays = []
    for (x_model, y_model) in keypoints_2d:
        u = x_model * sx
        v = y_model * sy
        ray = K_inv @ np.array([u, v, 1.0])
        rays.append(ray)
    return rays


@torch.no_grad()
def stage_corner_detection_dl(
    frame, depth, K, model, extract_keypoints_fn, train_cfg,
    model_variant, threshold, device,
):
    """
    Stage 2 — DL corner detection.

    Runs the learned heatmap model on the camera frame (+ depth if RGB-D),
    extracts peak keypoints via NMS, and backprojects them to camera-space
    rays so that Stage 3/4 can consume them identically to RANSAC corners.

    Returns list of np.ndarray (3,) rays in camera space.
    """
    print(f"\n=== Stage 2: DL Corner Detection  [{model_variant}] ===")

    if frame is None:
        print("  [!] No camera frame available — skipping DL detection.")
        return []

    image_size = train_cfg['image_size']
    orig_hw    = frame.shape[:2]   # (H, W) of original frame

    # Prepare inputs
    depth_np = depth if model_variant == 'dino_d_rgb' else None
    rgb_tensor, depth_tensor = _preprocess_for_dl(frame, depth_np, image_size, device)

    # Forward pass
    if model_variant == 'dino_d_rgb':
        raw_out = model(rgb_tensor, depth_tensor)   # [1, 1, H_map, W_map]
    else:
        raw_out = model(rgb_tensor)                 # [1, 1, H_map, W_map]

    heatmap   = torch.sigmoid(raw_out).squeeze(0)  # [1, H_map, W_map]
    kps_2d    = extract_keypoints_fn(heatmap, threshold=threshold, image_size=image_size)

    print(f"  Detected {len(kps_2d)} keypoint(s)  (threshold={threshold})")
    for i, (x, y) in enumerate(kps_2d):
        print(f"    Keypoint {i}: image [{x:.1f}, {y:.1f}] px  (model space)")

    # Backproject to camera-space rays
    corners_cam = _backproject_keypoints(kps_2d, orig_hw, image_size, K)

    for i, ray in enumerate(corners_cam):
        print(f"    Corner  {i}: cam-ray  [{ray[0]:+.4f}, {ray[1]:+.4f}, {ray[2]:+.4f}]")

    return corners_cam


def stage_corner_matching(corners_cam, calib, floor_result, fp_anns, map_h):
    """Stage 3 — unchanged from end2end_homography.py."""
    print("\n=== Stage 3: Corner Matching ===")

    if not corners_cam or not fp_anns:
        reason = "no detected corners" if not corners_cam else "no floorplan annotations"
        print(f"  [!] Skipping — {reason}.")
        return [], {}

    R_est, cam_world_est = build_estimated_pose(calib, floor_result)

    print(f"  Estimated pose:")
    print(f"    cam_world = [{cam_world_est[0]:.2f}, {cam_world_est[1]:.2f}, {cam_world_est[2]:.2f}]")
    if floor_result:
        print(f"    yaw={calib['yaw_deg']:.1f}°  "
              f"pitch={floor_result['pred_pitch']:.2f}°  "
              f"roll={floor_result['pred_roll']:.2f}°")
    else:
        print(f"    yaw={calib['yaw_deg']:.1f}°  pitch=0°  roll=0°  (Stage 1 unavailable)")

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
    print(f"  Matched {n_matched}/{len(matches)} corner(s)  (threshold: {MAX_MATCH_ANGLE_DEG}°)")

    for m in matches:
        pid  = m['point_id'] or "(no match)"
        err  = m['angular_err_deg']
        flag = "" if m['point_id'] else f"  [best was {err:.1f}° > {MAX_MATCH_ANGLE_DEG}°]"
        print(f"    Corner {m['corner_idx']} → {pid:8s}  angular err {err:.1f}°{flag}")

    return matches, ann_world


def _estimate_height_from_best_match(matches, corners_cam, ann_world, calib, R_est):
    """Identical to end2end_homography.py."""
    valid = [m for m in matches if m['point_id'] is not None]
    if not valid:
        return None, None, None

    best       = min(valid, key=lambda m: m['angular_err_deg'])
    corner_cam = np.array(corners_cam[best['corner_idx']], dtype=np.float64)

    P_ann  = ann_world[best['point_id']]
    cx, cy = float(calib['cam_xy'][0]), float(calib['cam_xy'][1])
    d_horiz = math.sqrt((P_ann[0] - cx) ** 2 + (P_ann[1] - cy) ** 2)
    if d_horiz < 1e-6:
        return None, None, None

    v_world     = R_est.T @ corner_cam
    d_rel_horiz = math.sqrt(v_world[0] ** 2 + v_world[1] ** 2)
    if d_rel_horiz < 1e-8:
        return None, None, None

    s     = d_horiz / d_rel_horiz
    h_est = float(-s * v_world[2])

    if h_est <= 0:
        return None, None, None

    return h_est, float(s), best


def _build_floor_homography(K, R, cam_world, calib, map_h):
    """Identical to end2end_homography.py."""
    cam_world = np.array(cam_world, dtype=np.float64)
    r1, r2   = R[:, 0], R[:, 1]
    t        = -R @ cam_world

    H_floor_to_img = K @ np.column_stack([r1, r2, t])

    scale, tx, ty = calib['scale'], calib['trans_x'], calib['trans_y']
    M_w2fp = np.array([
        [scale,   0.0,   scale * tx              ],
        [0.0,    -scale, map_h - scale * ty       ],
        [0.0,    0.0,   1.0                       ],
    ])

    return M_w2fp @ np.linalg.inv(H_floor_to_img)


def _compare_homographies(H_est, H_gt, cam_image_pts, fp_anns, scale_px_per_m):
    """Identical to end2end_homography.py."""
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
    """Stage 4 — identical to end2end_homography.py."""
    print("\n=== Stage 4: Homography ===")
    result = {'H_bev': None, 'H_fp': None, 'H_fp_gt': None, 'fp_anns': fp_anns}

    K = calib['K']

    world_az = direction_attr_to_world_azimuth(calib['yaw_deg'])
    pitch = floor_result['pred_pitch'] if floor_result else 0.0
    roll  = floor_result['pred_roll']  if floor_result else 0.0
    R_est = build_rotation_matrix(world_az, pitch, roll)

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

    cam_world_est = np.array([calib['cam_xy'][0], calib['cam_xy'][1], h_est])
    H_fp = _build_floor_homography(K, R_est, cam_world_est, calib, map_h)
    result['H_fp'] = H_fp
    print(f"  Estimated H   : yaw={calib['yaw_deg']:.1f}°  "
          f"pitch={pitch:.2f}°  roll={roll:.2f}°  h={h_est:.3f} m")

    H_fp_gt = _build_floor_homography(K, calib['R_gt'], calib['cam_world_gt'], calib, map_h)
    result['H_fp_gt'] = H_fp_gt
    print(f"  GT H          : built from GT extrinsic  (h={calib['height_gt']:.3f} m)")

    print("\n  Reprojection error  (image annotation → floorplan pixel):")
    _compare_homographies(H_fp, H_fp_gt, cam_image_pts, fp_anns, calib['scale'])

    return result


# ══════════════════════════════════════════════════════════════════════════════
# VISUALISATION  (identical to end2end_homography.py)
# ══════════════════════════════════════════════════════════════════════════════

def _draw_corners_on_frame(frame, corners_cam, K, color=(0, 255, 255)):
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
    out = floorplan.copy()
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
    out = frame.copy()
    h, w = out.shape[:2]

    ann_pts = {p['point_id']: (int(p['x']), int(p['y']))
               for p in cam_anns.get(camera_id, [])}

    proj = {}
    for i, corner in enumerate(corners_cam):
        u, v = project_3d_to_2d(corner, K)
        if 0 <= u < w and 0 <= v < h:
            proj[i] = (u, v)

    for m in matches:
        if m['point_id'] is None:
            continue
        pid, ci = m['point_id'], m['corner_idx']
        if pid in ann_pts and ci in proj:
            cv2.line(out, proj[ci], ann_pts[pid], (200, 200, 200), 2, cv2.LINE_AA)

    for ci, (u, v) in proj.items():
        cv2.circle(out, (u, v), 10, (0, 215, 255), -1)
        cv2.circle(out, (u, v), 10, (0, 0, 0), 2)
        cv2.putText(out, str(ci), (u + 13, v - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 215, 255), 2)

    for pid, (ax, ay) in ann_pts.items():
        cv2.circle(out, (ax, ay), 10, (255, 80, 0), -1)
        cv2.circle(out, (ax, ay), 10, (0, 0, 0), 2)
        cv2.putText(out, pid, (ax + 13, ay - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 80, 0), 2)

    lx, ly, ls = 14, h - 60, 0.55
    cv2.circle(out, (lx + 8, ly),      8, (0, 215, 255), -1)
    cv2.putText(out, "DL corner (predicted)", (lx + 22, ly + 5),
                cv2.FONT_HERSHEY_SIMPLEX, ls, (255, 255, 255), 1)
    cv2.circle(out, (lx + 8, ly + 26), 8, (255, 80, 0), -1)
    cv2.putText(out, "annotated point",       (lx + 22, ly + 31),
                cv2.FONT_HERSHEY_SIMPLEX, ls, (255, 255, 255), 1)

    return out


def save_outputs(output_dir, camera_id, frame, floorplan, corners_cam,
                 matches, calib, H_dict, map_h):
    os.makedirs(output_dir, exist_ok=True)

    def save(name, img):
        path = os.path.join(output_dir, f'{camera_id}_{name}')
        cv2.imwrite(path, img)
        print(f"  Saved: {path}")

    if frame is not None:
        if corners_cam:
            save('corners_dl.jpg',
                 _draw_corners_on_frame(frame, corners_cam, calib['K']))
        if matches:
            save('matches_dl.jpg',
                 _draw_matches_on_frame(frame, corners_cam, matches, calib['K']))

        if H_dict['H_bev'] is not None:
            bev = cv2.warpPerspective(frame, H_dict['H_bev'], BEV_SIZE)
            save('bev_dl.jpg', bev)

        if floorplan is not None:
            fp_w, fp_h = floorplan.shape[1], floorplan.shape[0]
            alpha = 0.5
            if H_dict['H_fp'] is not None:
                warped = cv2.warpPerspective(frame, H_dict['H_fp'], (fp_w, fp_h))
                blend  = cv2.addWeighted(floorplan, 1 - alpha, warped, alpha, 0)
                save('fp_warp_blend_dl.jpg', blend)
            if H_dict.get('H_fp_gt') is not None:
                warped_gt = cv2.warpPerspective(frame, H_dict['H_fp_gt'], (fp_w, fp_h))
                blend_gt  = cv2.addWeighted(floorplan, 1 - alpha, warped_gt, alpha, 0)
                save('fp_warp_blend_gt.jpg', blend_gt)

    if floorplan is not None and matches:
        save('floorplan_matches_dl.jpg',
             _draw_floorplan_matches(floorplan, matches, calib, H_dict['fp_anns'], map_h))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def run(scenario_dir, camera_id,
        depth_path=None, mask_path=None, annotations_dir=None, output_dir=None):

    depth_path, mask_path, annotations_dir, output_dir = _resolve_paths(
        scenario_dir, camera_id, depth_path, mask_path, annotations_dir, output_dir
    )

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f" End-to-End Homography (DL)  |  {camera_id}  |  {DL_MODEL}")
    print(f"{'='*60}")
    print(f" Scenario    : {scenario_dir}")
    print(f" Depth       : {depth_path}")
    print(f" Masks       : {mask_path or '(not found)'}")
    print(f" Annotations : {annotations_dir}")
    print(f" Checkpoint  : {DL_CHECKPOINT}")
    print(f" Output      : {output_dir}")

    # ── load DL model ────────────────────────────────────────────────────────
    model, train_cfg, extract_keypoints_fn = load_dl_model(
        DL_CHECKPOINT, DL_MODEL, device
    )

    # ── load data ────────────────────────────────────────────────────────────
    calib      = load_calibration(scenario_dir, camera_id)
    depth      = load_depth(depth_path)
    floor_mask, wall_mask = load_masks(mask_path)
    frame      = load_frame(scenario_dir, camera_id)
    floorplan  = load_floorplan(scenario_dir)
    fp_anns, cam_anns = load_annotations(annotations_dir)

    visible_ids = {p['point_id'] for p in cam_anns.get(camera_id, [])}
    fp_anns = {pid: v for pid, v in fp_anns.items() if pid in visible_ids}

    map_h = floorplan.shape[0] if floorplan is not None else 1080

    # ── pipeline ─────────────────────────────────────────────────────────────
    floor_result = stage_floor_geometry(depth, floor_mask, calib['K'], calib)

    corners_cam = stage_corner_detection_dl(
        frame, depth, calib['K'],
        model, extract_keypoints_fn, train_cfg,
        DL_MODEL, DL_THRESHOLD, device,
    )

    matches, ann_world = stage_corner_matching(
        corners_cam, calib, floor_result, fp_anns, map_h
    )

    if frame is not None:
        results_dir = os.path.join(_ROOT, 'results')
        os.makedirs(results_dir, exist_ok=True)
        stage3_img  = _draw_stage3_matching(
            frame, corners_cam, matches, cam_anns, camera_id, calib['K']
        )
        stage3_path = os.path.join(results_dir, f'{camera_id}_stage3_matching_DL.jpg')
        cv2.imwrite(stage3_path, stage3_img)
        print(f"\n  Stage 3 visualisation saved: {stage3_path}")

    H_dict = stage_homography(
        corners_cam, matches, floor_result, calib,
        ann_world, fp_anns,
        cam_anns.get(camera_id, []),
        map_h,
    )

    save_outputs(output_dir, camera_id, frame, floorplan,
                 corners_cam, matches, calib, H_dict, map_h)

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