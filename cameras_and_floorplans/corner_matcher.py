"""
Corner Matcher
--------------
Matches detected 3D corners (camera space) to annotated world-space corners
by 3D ray-direction comparison.

All inputs are estimated or independently known — no GT extrinsic, no scale factor.

Caller responsibilities (done once in end2end_homography.py, not here):
  • Convert 'direction' attribute → world azimuth via direction_attr_to_world_azimuth()
  • Build R via build_rotation_matrix(world_azimuth, pitch_deg, roll_deg)
  • Build cam_world = [coord_x, coord_y, estimated_height]
  • Convert annotated floorplan pixels → world 3D with scale (the only use of scale)

This module is then purely geometric: R, cam_world, K, ann_world — no calibration access.
"""

import math
import numpy as np


# ── angle convention helper ───────────────────────────────────────────────────

def direction_attr_to_world_azimuth(direction_deg):
    """
    Convert the dataset 'direction' attribute to world azimuth in radians.

    The attribute uses a screen-space convention (Y down, -90° offset).
    World azimuth is CCW from +X in world XY (world Y up).

    Verified: direction=321.86° → world azimuth ≈ 128.14°, matching Camera_0000 GT R.
    """
    yaw_screen = math.radians(direction_deg - 90)
    return math.atan2(-math.sin(yaw_screen), math.cos(yaw_screen))


# ── rotation builder ──────────────────────────────────────────────────────────

def build_rotation_matrix(world_azimuth_rad, pitch_deg, roll_deg=0.0):
    """
    Build the world-to-camera rotation matrix from estimated camera angles.

    Parameters
    ----------
    world_azimuth_rad : direction camera faces — CCW from +X in world XY (world Y up).
                        Obtain from direction_attr_to_world_azimuth(direction_deg).
    pitch_deg         : downward tilt. Positive = camera looks at the floor.
                        Estimated from the floor normal (Stage 1).
    roll_deg          : right-side-down tilt. 0 for a level camera.
                        Estimated from the floor normal (Stage 1).

    Returns
    -------
    R : (3,3) ndarray — world-to-camera rotation (P_cam = R @ P_world + t).

    Convention verified: R built from GT angles matches GT extrinsic R to floating-point precision.
    """
    theta = world_azimuth_rad
    phi   = math.radians(pitch_deg)
    rho   = math.radians(roll_deg)

    ct, st = math.cos(theta), math.sin(theta)
    cp, sp = math.cos(phi),   math.sin(phi)
    cr, sr = math.cos(rho),   math.sin(rho)

    # Camera axes in world for zero roll
    X0 = np.array([st, -ct, 0.0])          # camera right (horizontal)
    Z0 = np.array([ct*cp, st*cp, -sp])     # camera forward (pitching down reduces Z world-component)
    Y0 = np.cross(Z0, X0)                  # camera down  = forward × right

    # Apply roll: rotate X0, Y0 around Z0 by rho
    X = cr * X0 + sr * Y0
    Y = -sr * X0 + cr * Y0

    # R_cam_to_world columns = [X, Y, Z0]; world-to-camera = transpose
    return np.column_stack([X, Y, Z0]).T


# ── core matching ─────────────────────────────────────────────────────────────

def _angle_deg(v1, v2):
    """Angular difference in degrees between two direction vectors."""
    c = float(np.clip(
        np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)),
        -1.0, 1.0
    ))
    return math.degrees(math.acos(c))


def match_corners(
    corners_cam,
    R,
    cam_world,
    ann_world,
    max_angle_deg=15.0,
):
    """
    Match detected 3D corners (camera space) to annotated world corners by ray direction.

    Scale-invariant: only directions are compared, not distances.
    No GT extrinsic, no scale factor used inside this function.

    Parameters
    ----------
    corners_cam   : list of np.ndarray (3,) — detected corners in camera space,
                    from extract_room_corners(). Depth scale does not matter.
    R             : (3,3) world-to-camera rotation — from build_rotation_matrix().
    cam_world     : (3,) camera world position [coord_x, coord_y, estimated_height].
                    coord_x/y from calibration coordinates field (NOT from extrinsic);
                    height from Stage 1 floor-normal estimation.
    ann_world     : {point_id: np.ndarray (3,)} annotated corners in world space (Z=0).
                    Caller converts from floorplan pixels using scale — once, externally.
    max_angle_deg : reject a match if its angular error exceeds this threshold.

    Returns
    -------
    list of dicts, one per corner:
        corner_idx      : index into corners_cam
        point_id        : matched annotation ID, or None if no match within threshold
        angular_err_deg : angle between detected-corner ray and matched-annotation direction
    """
    R_cw = R.T  # camera-to-world = R^T

    results = []
    for i, P_cam in enumerate(corners_cam):
        norm = np.linalg.norm(P_cam)
        if norm < 1e-8:
            continue

        d_world = R_cw @ (P_cam / norm)   # direction in world space

        best_id, best_err = None, float('inf')
        for pid, P_ann in ann_world.items():
            d_ann = P_ann - cam_world
            if np.linalg.norm(d_ann) < 1e-6:
                continue
            err = _angle_deg(d_world, d_ann)
            if err < best_err:
                best_err = err
                best_id  = pid

        results.append({
            'corner_idx':      i,
            'point_id':        best_id if best_err <= max_angle_deg else None,
            'angular_err_deg': best_err,
        })

    return results


def match_pixel(u, v, K, R, cam_world, ann_world, max_angle_deg=15.0):
    """
    Match a single detected corner pixel (u, v) to the nearest annotated corner.

    Unprojects to a ray using K, then delegates to match_corners.
    Same parameter semantics as match_corners.
    """
    ray_cam = np.linalg.inv(K) @ np.array([float(u), float(v), 1.0])
    return match_corners([ray_cam], R, cam_world, ann_world, max_angle_deg)[0]


# ── homography pair builder ───────────────────────────────────────────────────

def build_homography_pairs(matches, corners_cam, K, ann_pixels):
    """
    Build parallel point arrays for cv2.findHomography from match results.

    No scale or world coordinates needed — source points are reprojected camera
    image pixels; destination points are floorplan pixels from the annotation dict.

    Parameters
    ----------
    matches     : output of match_corners
    corners_cam : same list of 3D camera-space corners passed to match_corners
    K           : (3,3) intrinsic — reprojects 3D corners to camera image pixels
    ann_pixels  : {point_id: {x, y}} — original floorplan pixel annotations
                  (NOT the world-converted dict passed to match_corners)

    Returns
    -------
    src_pts : (N,2) float32 — camera image pixels (u, v)
    dst_pts : (N,2) float32 — floorplan image pixels (fp_x, fp_y)
    n_pairs : int
    """
    src, dst = [], []
    for m in matches:
        if m['point_id'] is None:
            continue
        P_cam = corners_cam[m['corner_idx']]
        pp = K @ P_cam
        if abs(pp[2]) < 1e-8:
            continue
        src.append([pp[0] / pp[2], pp[1] / pp[2]])
        ann = ann_pixels[m['point_id']]
        dst.append([ann['x'], ann['y']])

    if not src:
        return None, None, 0

    return np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32), len(src)