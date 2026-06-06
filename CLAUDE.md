# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a computer vision thesis project on **camera-to-floorplan localization** for Multi-Target Multi-Camera (MTMC) tracking. The goal is to estimate camera geometry (floor normal, camera height) from surveillance video and project the scene into a 2D bird's-eye view (BEV) that aligns with a building floorplan.

Dataset: **Hospital_000_Val** from AI-SmartSpaces / MTMC_Tracking_2025 (8 cameras with calibration, ground truth, and video).

## Running Scripts

The project uses `uv` for dependency management:

```bash
uv run python <script.py>          # run any script
uv add <package>                   # add dependency
```

All scripts are standalone with hardcoded paths in `if __name__ == "__main__"` blocks. There are no CLI arguments — edit the config section at the top of each script before running.

GPU is auto-detected via `torch.cuda.is_available()`. All neural models (Mask2Former, DepthAnything) load onto GPU when available.

## Pipeline Architecture

The pipeline is sequential. Each stage outputs files consumed by the next.

### Stage 1 — Dataset Preparation (`dataset/`)
- `AI-SmartSpaces_download.py` — downloads the Hospital dataset from HuggingFace
- `AI-SmartSpaces_processing.py` — extracts the first frame from each video and writes per-camera metadata JSON (contains `calibration.intrinsicMatrix` and `calibration.extrinsicMatrix`)
- Output: `dataset/Point_Detection_Tests/Camera_XX_frame.jpg` + `Camera_XX_metadata.json`

### Stage 2 — Temporal Segmentation (`segmentation/`)
- `mask2former_video.py` — `TemporalVideoSegmenter`: samples video frames at a fixed interval, runs Mask2Former (ADE20K semantic: wall=0, floor=3, door=14), accumulates masks via logical OR across the video to produce a static background
- Output: `segmentation/temporal_masks/Camera_XX_temporal_bg.jpg` — color-coded image (green=floor, red=wall, cyan=door in BGR)
- `boundary_extraction.py` — isolates wall/floor regions from temporal masks via HSV thresholding, finds the wall-base boundary with dilation+AND, fits lines with Probabilistic Hough, merges collinear fragments
- Output: `segmentation/boundary_lines_colinear/` + `geometry/extracted_2d_lines.json`

### Stage 3 — Temporal Depth (`depth/`)
- `depth_anything_simp.py` — `TemporalDepthSegmenter`: infers depth on the first and last video frames only, aggregates using proportional thresholding (15% change = structural reveal) + EMA smoothing
- **Critical:** DepthAnything outputs *disparity* (inverse depth). The code normalizes disparity then inverts it (`depth = 1 / (disp_normalized + 0.05)`) to obtain relative depth suitable for 3D reconstruction
- Output: `depth/temporal_depth2/Camera_XX_temporal_depth_raw.npy` (float32) + colored visualization JPG

### Stage 4 — Wall Normal & Camera Height Estimation (`geometry/`)
- `wall_normals_dbscan.py` — `WallNormalExtractor`: loads depth `.npy` + wall mask from temporal_masks, creates Open3D point cloud, estimates surface normals, clusters on the Gaussian sphere with DBSCAN (eps=0.05 ≈ 2.9°) to find dominant wall directions
- `normals.py` — floor normal estimation (similar approach for floor)
- Output: `geometry/walls_normals/dominant_wall_normals.json`, `geometry/camera_extrinsics.json` (predicted floor normals + camera heights)

### Stage 5 — BEV Projection (`geometry/bev_projection.py`)
- `create_bev_homography(K, normal, height)` — builds a 3×3 homography from camera intrinsics, predicted floor normal, and camera height that warps the perspective image to a 16m×16m top-down view
- Warps both RGB and mask images; uses `INTER_NEAREST` for masks to preserve class colors
- Output: `geometry/bev_projections/Camera_XX_bev_rgb.jpg` + `_bev_mask.png`

### Stage 6 — Room Corner Detection (`geometry/corners_extraction.py`)
- `extract_room_corners(depth_map, floor_mask, wall_mask, K)` — unprojects masked depth pixels to 3D, fits floor plane with RANSAC, fits up to 4 wall planes iteratively, intersects floor+wall+wall triples to find 3D room corners
- Filters intersections: walls must be roughly perpendicular (dot product < 0.5) and corner must be in front of camera (Z > 0)

### Stage 7 — Floorplan Annotation (`cameras and floorplan/annotator.py`)
- `CameraAnnotator` — interactive matplotlib tool: displays floorplan map (left) and camera image (right), user clicks corresponding points, saves pairs to `annotated_floorplan.json` / `annotated_cameras.json`
- Navigation: arrow keys to switch cameras, ESC to cancel pending click

### Evaluation (`geometry/compare_groundtruth.py`)
- Compares predicted floor normals and camera heights against ground-truth extrinsic matrices
- GT camera position: `-R^T @ t` from the extrinsic matrix; GT normal: 3rd column of R
- Metrics: angular error (degrees), height error, suggested scale factor
- Output: `geometry/evaluation_summary.json`

## Key Conventions

**Calibration data structure** (from `calibration.json`):
```python
K = np.array(meta["calibration"]["intrinsicMatrix"])   # 3x3
ext = np.array(meta["calibration"]["extrinsicMatrix"]) # 3x4
R, t = ext[:, :3], ext[:, 3]
camera_position = -np.dot(R.T, t)   # world-space camera center
camera_height = abs(camera_position[2])
```

**Depth vs. disparity:** DepthAnything outputs disparity. After normalization to [0,1], use `depth = 1 / (disp + 0.05)`. Raw `.npy` files saved by `depth_anything_simp.py` are already converted to depth (not disparity).

**Mask color encoding** (BGR, for HSV extraction downstream):
- Floor: green `[0, 255, 0]`
- Wall: red `[0, 0, 255]`
- Door: cyan `[255, 255, 0]`

**Floor normal convention:** the normal is negated if `n[1] > 0` so it always points upward in camera space.
