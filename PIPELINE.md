# Homography Estimation Pipeline — Mathematical Overview

## High-Level Goal

The objective is to compute a **3×3 projective homography** `H` that maps any pixel `(u, v)` in a surveillance camera image to its corresponding location `(fp_x, fp_y)` on a 2D building floorplan. Because the floor is a planar surface in the scene, a single homography suffices: any point on the floor plane has a unique and consistent projection into both the camera image and the top-down floorplan.

The pipeline operates in two parallel tracks:
- **Classical geometry track** — depth + segmentation → 3D geometry → homography
- **Deep learning track** — learned corner detection → homography via matched point pairs

---

## Part 1 — Temporal Segmentation (`segmentation/mask2former_video.py`)

### What it does

Mask2Former is run on multiple video frames sampled at 1-second intervals. It predicts a per-pixel semantic class label using the ADE20K vocabulary. Three classes are extracted:

| Class | ADE20K index |
|---|---|
| Wall | 0 |
| Floor | 3 |
| Door | 14 |

### Temporal aggregation — logical OR

For each class, a binary mask is accumulated across frames using a **logical OR**:

```
accumulated_mask = accumulated_mask OR current_mask
```

The intuition: a walking person may temporarily occlude part of the floor or wall. Over many frames, every pixel that was ever seen as floor will eventually be revealed. The accumulated mask is thus a **temporal union** of the static background structure.

### Priority resolution

When classes overlap in the union, a strict priority hierarchy resolves conflicts:

```
Door  beats Wall   (a door opening is inside a wall — reveals background)
Floor beats Door   (the floor-door boundary is walkable)
Floor beats Wall   (a wall does not lie flat)
```

Applied in sequence by zeroing out lower-priority masks wherever higher-priority ones are active.

### Output encoding

A color-coded BGR image:

| Region | BGR value |
|---|---|
| Floor | `[0, 255, 0]` (green) |
| Wall | `[0, 0, 255]` (red) |
| Door | `[255, 255, 0]` (cyan) |

Downstream binary mask extraction uses HSV thresholding on this color image.

---

## Part 2 — Temporal Depth Estimation (`depth/depth_anything_simp.py`)

### The DepthAnything model

DepthAnything is a monocular depth estimation model based on a Vision Transformer backbone, trained with a scale-and-shift invariant loss (following MiDaS). Its output is **affine-ambiguous disparity** — not metric depth. Specifically it predicts a value monotonically related to `1/Z`, where `Z` is the true scene depth.

### Temporal aggregation — two frames with EMA

Only the **first and last frames** of the video are used. This captures the maximum temporal spread while minimizing compute. For each pixel, let `d_acc` be the accumulated disparity and `d_new` be the current frame's disparity:

```
proportional_diff = (d_acc - d_new) / (d_acc + ε)
```

Three cases:

| Condition | Interpretation | Action |
|---|---|---|
| `proportional_diff > 0.15` | Occluder was present, background now revealed | `d_acc = d_new` (accept outright) |
| `|proportional_diff| ≤ 0.15` | Small noise fluctuation | `d_acc = (1−α)·d_acc + α·d_new` with α=0.1 (EMA) |
| `proportional_diff < −0.15` | New occluder walked in | Do nothing (preserve background) |

### Disparity → relative depth

After aggregation:

```
disp_normalized = (d_acc − min(d_acc)) / (max(d_acc) − min(d_acc))
depth = 1 / (disp_normalized + 0.05)
```

The `+0.05` offset prevents division by zero for far pixels (near-zero disparity after normalization). The result is a **relative depth** map in unitless space: near pixels have large values, far pixels have small values. This is geometrically consistent for extracting surface normals (which require only direction, not scale), but not metric — camera height will be in relative units unless a scale factor is recovered later.

---

## Part 3 — The Pinhole Camera Model and Unprojection

All classical geometry stages share the same fundamental unprojection step.

### The pinhole model

A camera maps a 3D point `P = [X, Y, Z]` (in camera coordinates) to a 2D pixel `[u, v]` via the intrinsic matrix `K`:

```
[u]         [fx   0   cx] [X]
[v]  = (1/Z)[ 0  fy   cy] [Y]
[1]         [ 0   0    1] [Z]
```

Where `fx, fy` are focal lengths in pixels and `cx, cy` is the principal point. **OpenCV convention**: camera X points right, Y points **down**, Z points forward into the scene.

### Unprojection (inverse mapping)

Given depth value `Z` at pixel `(u, v)`, the 3D point in camera space is:

```
X = (u − cx) · Z / fx
Y = (v − cy) · Z / fy
Z = Z
```

Applied elementwise across all masked pixels to produce a dense point cloud. The floor mask and wall mask restrict which pixels contribute to each respective point cloud.

---

## Part 4 — Floor Plane Fitting and Normal Estimation (`geometry/normals.py`)

### Point cloud preparation

Floor pixels are unprojected to 3D. A voxel grid downsampling is applied with:

```
voxel_size = 0.01 × median_Z
```

This threshold is proportional to median depth so the same code works for both metric GT depth (in metres) and relative DepthAnything depth (unitless).

### RANSAC plane fitting

RANSAC iteratively:
1. Randomly samples 3 points from the floor point cloud
2. Fits a plane through those 3 points
3. Counts inliers — points within `distance_threshold = 0.01 × median_Z` of the plane
4. Keeps the model with the most inliers across 1000 iterations

A plane in 3D is:

```
aX + bY + cZ + d = 0
```

where `[a, b, c]` is the unit plane normal and `d` is the signed distance from the origin. Open3D's `segment_plane()` returns `[a, b, c, d]` normalized to `‖(a, b, c)‖ = 1`.

### Normal direction convention

In OpenCV camera space, Y points down. The floor normal should point "up" toward the ceiling, meaning its Y component must be **negative**. If `n[1] > 0`, both are negated:

```python
if normal[1] > 0:
    normal = -normal
    d = -d
```

### Camera height

The plane equation `aX + bY + cZ + d = 0` tells us that the signed distance from the origin (the camera center) to the plane is `|d|` (since the normal is unit-length). Since the camera is above the floor:

```
camera_height = |d|
```

For metric depth this is in metres. For relative depth this must be rescaled using a known reference distance (see Part 7).

### Pitch and roll from the floor normal

Given floor normal `n = [nx, ny, nz]` (pointing up, so `ny < 0`):

```
pitch = atan2(−nz, −ny)    [degrees]
roll  = atan2( nx, −ny)    [degrees]
```

When the camera is perfectly horizontal: `n = [0, −1, 0]` → pitch = 0, roll = 0. Pitch is positive when the camera tilts downward (typical for ceiling-mounted surveillance cameras). Roll is positive when the right side is lower.

---

## Part 5 — BEV Homography from Floor Normal

### Camera rotation from yaw, pitch, roll

The yaw angle (azimuth — which direction the camera faces horizontally) is read from the calibration `direction` attribute. The dataset uses a screen-space convention (Y down, −90° offset), converted to world azimuth as:

```
yaw_screen    = radians(direction_deg − 90)
world_azimuth = atan2(−sin(yaw_screen), cos(yaw_screen))
```

From yaw `θ`, pitch `φ`, and roll `ρ`, the world-to-camera rotation matrix `R` is:

```
Camera axes in world space (zero roll):
  X₀ = [sin θ, −cos θ, 0]                       (camera right — horizontal)
  Z₀ = [cos θ·cos φ, sin θ·cos φ, −sin φ]       (camera forward)
  Y₀ = Z₀ × X₀                                  (camera down = forward × right)

Apply roll ρ around Z₀:
  X =  cos ρ · X₀ + sin ρ · Y₀
  Y = −sin ρ · X₀ + cos ρ · Y₀

R = [X | Y | Z₀]ᵀ     (world-to-camera; P_cam = R · P_world + t)
```

This construction is verified: when GT angles are used, the resulting `R` matches the GT extrinsic rotation to floating-point precision.

### Analytical floor-to-floorplan homography

The floorplan is a 2D metric map with scale `s` (pixels per metre) and offset `(tx, ty)`. The world-to-floorplan pixel mapping is:

```
fp_x =  s · (world_x + tx)
fp_y =  map_H − s · (world_y + ty)
```

Captured as a 3×3 matrix:

```
         ⎡ s    0    s·tx         ⎤
M_w2fp = ⎢ 0   −s    map_H − s·ty ⎥
         ⎣ 0    0    1            ⎦
```

For a point on the floor plane (world Z = 0), the camera image pixel and world XY position are related by the **plane-induced homography**:

```
H_floor_to_img = K · [r₁ | r₂ | t]
```

where `r₁`, `r₂` are the first two columns of `R` (world X and Y axes in camera space) and `t = −R · cam_world`. This encodes the projective relationship between the floor plane (parameterized by (X, Y, 0)) and the camera image.

The complete image-to-floorplan homography is:

```
H_img_to_fp = M_w2fp · inv(K · [r₁ | r₂ | t])
```

To map camera pixel `(u, v)` to floorplan pixel `(fp_x, fp_y)`:

```
[fp_x']         [u]
[fp_y'] = H_img_to_fp · [v]
[ w   ]         [1]

fp_x = fp_x' / w
fp_y = fp_y' / w
```

### The scale problem with relative depth

When DepthAnything is used, `cam_world[2]` (the height) is in relative units. The homography is geometrically correct in **direction** but wrong in scale. This is resolved in Part 7 by recovering a metric scale factor from a single matched corner with a known world distance.

---

## Part 6 — Wall Planes and 3D Corner Detection (`geometry/corners_extraction.py`)

### Wall point cloud and surface normal estimation

Wall pixels are unprojected to 3D. Open3D's `estimate_normals()` is called with a KD-tree hybrid search (`radius = 0.05 × median_Z`, max 30 neighbors). This computes per-point surface normals by fitting a local PCA covariance matrix and taking the eigenvector corresponding to the smallest eigenvalue.

### Iterative RANSAC wall plane extraction

Up to 4 wall planes are extracted sequentially:

1. Run RANSAC on the remaining wall point cloud → dominant plane model `[a, b, c, d]`
2. Remove inlier points from the cloud (`select_by_index(..., invert=True)`)
3. If the new plane's normal is nearly parallel to an already-found plane (`n_new · n_old > 0.95`), discard — prevents two faces of the same corridor wall from generating a false corner
4. Otherwise, add the plane model and repeat

`distance_threshold = 0.05 × median_Z` is deliberately **looser** here than for the floor (which uses 1%). Walls are noisier, and a tight threshold causes RANSAC to find micro-patches on a single wall surface and burn all iterations without finding distinct walls.

### 3-plane intersection

A room corner is the intersection of the floor plane and two perpendicular wall planes. Three planes:

```
a₁X + b₁Y + c₁Z = −d₁
a₂X + b₂Y + c₂Z = −d₂
a₃X + b₃Y + c₃Z = −d₃
```

define a unique intersection point found by solving the 3×3 linear system:

```
N · P = −D

    ⎡ a₁  b₁  c₁ ⎤          ⎡ d₁ ⎤
N = ⎢ a₂  b₂  c₂ ⎥    D =  ⎢ d₂ ⎥
    ⎣ a₃  b₃  c₃ ⎦          ⎣ d₃ ⎦
```

via `numpy.linalg.solve`. A singular matrix (degenerate geometry) produces no corner.

### Geometric validity filters

Two filters validate each candidate corner:

1. **Perpendicularity**: the two wall normals must be roughly perpendicular — `|n_a · n_b| < 0.5` (angle > ~60°). Rejects corners formed by near-parallel walls.
2. **In front of camera**: the corner must have `Z > 0` in camera space.

---

## Part 7 — Corner Matching and Metric Height Recovery

### The matching problem

After detecting 3D room corners in camera space, each must be associated with a specific annotated point on the floorplan. Annotations are 2D pixel coordinates on `map.png` manually labeled to correspond to physical room corners.

### Scale invariance

With relative depth, the detected corner position `P_cam = [X, Y, Z]` is in unknown units. However, the **direction** of the ray from the camera to the corner is scale-invariant. This is the key insight enabling matching without knowing the depth scale.

### Ray direction matching (`cameras_and_floorplans/corner_matcher.py`)

For each detected corner `P_cam`:

1. Normalize: `d_cam = P_cam / ‖P_cam‖`
2. Rotate to world space: `d_world = Rᵀ · d_cam` (R is orthogonal, so `R⁻¹ = Rᵀ`)
3. For each annotated floorplan point `P_ann` (world coordinates, Z=0):
   ```
   d_ann = (P_ann − cam_world) / ‖P_ann − cam_world‖
   θ = arccos(d_world · d_ann)
   ```
4. Assign the annotation with minimum `θ` if `θ < MAX_MATCH_ANGLE_DEG = 4.5°`

The camera world position `cam_world = [coord_x, coord_y, h_est]` uses `coordinates.x/y` from the calibration (independent of the GT extrinsic) and the estimated height from Stage 1.

### Converting floorplan pixels to world coordinates

The inverse of the world→floorplan formula:

```
world_x = fp_x / s − tx
world_y = (map_H − fp_y) / s − ty
world_z = 0
```

This conversion uses `scaleFactor` and `translationToGlobalCoordinates` from the calibration, applied exactly once before calling `match_corners()`.

### Metric height recovery from the best match

When depth is relative, `P_cam` is off by an unknown scale factor `s`. The best-matched corner gives a known horizontal distance in world space:

```
d_horiz = ‖P_ann[:2] − cam_world[:2]‖     (metres, from floorplan)
```

In relative units, the same direction in world space gives:

```
v_world      = Rᵀ · P_cam
d_rel_horiz  = ‖v_world[:2]‖
```

Since `s · d_rel_horiz = d_horiz`:

```
s = d_horiz / d_rel_horiz
```

The metric camera height is then:

```
h_metric = −s · v_world[2]
```

(`v_world[2]` is negative for a floor corner — the floor is below the camera in world Z.) This recovered metric height is substituted back into the analytical homography to produce a metrically-correct `H_img_to_fp`.

---

## Part 8 — Deep Learning Track: Heatmap Regression

Rather than deriving corners from depth and geometry, the DL track trains a network to directly predict a **probability heatmap** over the image where each room corner (floor-wall intersection) is likely to be located.

### Ground truth: Gaussian heatmaps

Each annotated keypoint `(kx, ky)` in heatmap-space generates a 2D Gaussian blob:

```
H_gt(x, y) = exp( −((x − kx)² + (y − ky)²) / (2σ²) )
```

with `σ = 2.0` heatmap pixels. When multiple keypoints overlap, the element-wise maximum is taken — each peak reaches exactly 1.0. The resulting heatmap `H_gt ∈ [0, 1]` has values close to 1 at corner locations and decays to 0 elsewhere.

Heatmap resolution: `(image_size / patch_size) × 2^N = 16×16 × 4 =` **64×64** for a 224-pixel input with 2 decoder blocks.

### Loss function: CenterNet penalty-reduced focal loss

Standard BCE would be dominated by the overwhelming number of non-corner pixels. The **CenterNet focal loss** addresses this:

Let `p = sigmoid(pred)` and `gt ∈ [0, 1]` be the target heatmap.

```
Positive term (gt == 1.0 only):
  L_pos = −(1 − p)^α · log(p)

Negative term (gt < 1.0):
  L_neg = −(1 − gt)^β · p^α · log(1 − p)

Total:
  L = −(Σ L_pos + Σ L_neg) / N_pos
```

With `α = 2, β = 4`:

- `(1 − p)^α` in `L_pos` focuses training on hard positives (underconfident peaks)
- `(1 − gt)^β` in `L_neg` **suppresses the penalty near Gaussian peaks** — pixels within the Gaussian spread are not strongly penalized for having non-zero predictions, since they genuinely are near a corner

### Inference: NMS via MaxPool

After sigmoid, peaks are extracted using **2D Non-Maximum Suppression** via 3×3 max pooling:

```python
pooled = max_pool2d(heatmap, kernel_size=3, stride=1, padding=1)
keep   = (heatmap == pooled) AND (heatmap >= threshold)
```

A pixel survives only if it equals the local 3×3 maximum AND exceeds `threshold = 0.3`. Surviving pixel coordinates are scaled back to image space.

---

## Part 9 — Model Architectures

### A: RGB-only — `dino_rgb/training.py` (Baseline)

```
Input: [B, 3, 224, 224]   (ImageNet-normalised RGB)

A. Frozen DINOv2-S backbone:
     → patch tokens  [B, 256, 384]   (256 patches of 14×14, 384-dim embeddings)
     → reshape to    [B, 384, 16, 16]

B. Neck (1×1 Conv + BN + ReLU):
     [B, 384, 16, 16]  →  [B, 256, 16, 16]

C. Decoder (2× UpsampleBlock: 2× bilinear + Conv 3×3 + BN + ReLU):
     [B, 256, 16, 16]  →  [B, 128, 32, 32]  →  [B, 64, 64, 64]

D. Head (1×1 Conv, raw logits):
     [B, 64, 64, 64]   →  [B, 1, 64, 64]
```

DINOv2 is completely frozen. Only B, C, D are trained (AdamW + cosine LR schedule). This is the ablation baseline — it tests how much a purely semantic/textural representation can locate corners without explicit depth.

### B: RGB-D — `dino_d_rgb/training.py` (Late Fusion)

```
RGB stream:
  [B, 3, 224, 224]  →  frozen DINOv2-S  →  [B, 384, 16, 16]

Depth stream (trainable ResNet18, modified):
  conv1: 3-ch → 1-ch  (new weights = sum of original 3-ch filters,
                        preserves edge-detection structure)
  stem → layer1 → layer2 → layer3 → layer4 → adaptive_avg_pool2d
  [B, 1, 224, 224]  →  [B, 512, 16, 16]

Late fusion:
  concat([rgb_feat, depth_feat])  →  [B, 896, 16, 16]

Neck (1×1 Conv + BN + ReLU):
  [B, 896, 16, 16]  →  [B, 256, 16, 16]

Decoder + Head (identical to RGB-only):
  →  [B, 1, 64, 64]
```

Geometric augmentation (flips, crops) is applied to both streams identically using `albumentations.ReplayCompose` — the exact same transform is replayed on the depth map to keep them spatially aligned. ColorJitter is applied to RGB only (depth is illumination-invariant).

### C: KeypointDETR — `detr/train.py` (Set Prediction)

Rather than a dense heatmap, DETR poses corner detection as a **set prediction problem**. A fixed number of query slots (e.g., 100) each predict one keypoint `(x, y, visibility)`. The loss uses **Hungarian matching** to find the optimal one-to-one assignment between predicted and ground-truth keypoints, minimizing total `L1` distance in normalized coordinates. Unmatched predictions are assigned to a no-object class. The standard bounding-box head is replaced with a 3-layer keypoint MLP.

---

## Part 10 — Full Estimation Chain

```
Video
  │
  ├─ Mask2Former (temporal OR)       →  floor_mask, wall_mask
  │
  └─ DepthAnything (first + last frame)
       disparity  →  normalize  →  invert  →  depth_map
                                                    │
                                       ┌────────────┴────────────┐
                             Classical │                          │ Deep learning
                             geometry  │                          │
                                       │                          │
               floor_mask + depth_map + K              DINOv2 / DETR
                             │                                    │
                    Unproject to 3D                    →  2D keypoints (u, v)
                             │
                    RANSAC on floor points
                             │
                    floor normal n,  height h (relative)
                             │
                    pitch = atan2(−nz, −ny)
                    roll  = atan2( nx, −ny)
                             │
                    R  from  yaw + pitch + roll
                             │
               wall_mask + depth_map + K
                             │
                    Iterative RANSAC on walls
                             │
                    3-plane intersection  →  3D corners P_cam
                             │
                    Ray-direction matching vs. floorplan annotations
                    (θ = arccos(Rᵀ·P_cam/‖…‖  ·  d_ann),  threshold 4.5°)
                             │
                    Metric scale recovery:
                      s        = d_horiz / d_rel_horiz
                      h_metric = −s · (Rᵀ · P_cam)[2]
                             │
                             └──────────────┬──────────────────────┘
                                            │
                  H = M_w2fp · inv( K · [r₁ | r₂ | t] )

                  where  t = −R · [coord_x, coord_y, h_metric]ᵀ

                                            │
                               cv2.warpPerspective(frame, H, (map_W, map_H))
                                            │
                                   Floorplan-aligned output
```

---

## Part 11 — The GT-Free Estimation Principle

A key architectural constraint throughout: **the extrinsic matrix from the calibration file is never used during estimation.** The full pose is assembled from:

| Source | Field used | Purpose |
|---|---|---|
| `calibration.json` | `coordinates.x/y` | Camera world XY position |
| `calibration.json` | `attributes[direction]` | Yaw (via `direction_attr_to_world_azimuth`) |
| `calibration.json` | `scaleFactor`, `translationToGlobalCoordinates` | World↔floorplan pixel conversion |
| Stage 1 (floor plane) | `n`, `h` | Pitch, roll, relative height |
| Stage 3 (corner match) | Matched annotation | Metric depth scale |

The GT extrinsic (`extrinsicMatrix`) is loaded only for quantitative error reporting — computing angular error on the floor normal and height error — and is never passed to any estimation function. This makes the method applicable to cameras where only intrinsics and rough map annotations are known.

---

## Summary of Key Mathematical Facts

| Quantity | How derived | Formula |
|---|---|---|
| 3D point from pixel | Pinhole unprojection | `X=(u−cx)Z/fx`, `Y=(v−cy)Z/fy` |
| Depth from disparity | Invert normalized disparity | `depth = 1 / (disp_norm + 0.05)` |
| Floor normal | RANSAC on floor point cloud | `argmax inliers of  aX+bY+cZ+d=0` |
| Camera height | Plane-to-origin distance | `h = |d|` |
| Pitch | From floor normal | `atan2(−nz, −ny)` |
| Roll | From floor normal | `atan2(nx, −ny)` |
| Camera rotation R | Yaw + pitch + roll | `[X\|Y\|Z₀]ᵀ` |
| 3D corner | Solve 3-plane linear system | `N·P = −D` |
| Corner match | Ray direction angle | `arccos(Rᵀ·d_cam · d_ann)` |
| Depth scale factor | Known horizontal distance | `s = d_horiz / d_rel_horiz` |
| Metric height | Scale × relative height | `h = −s · (Rᵀ·P_cam)[2]` |
| Floor homography | Plane-induced projective map | `M_w2fp · inv(K·[r₁\|r₂\|t])` |
| GT heatmap | Gaussian blob per keypoint | `exp(−(Δx²+Δy²) / 2σ²)` |
| Focal loss weighting | Penalty reduction near peaks | `(1−gt)^β · p^α · log(1−p)` |
| Corner NMS | Local max pooling | `H == MaxPool₃ₓ₃(H)  AND  H ≥ θ` |