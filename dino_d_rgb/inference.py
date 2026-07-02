#!/usr/bin/env python3
"""
inference.py — RGB-D Room Corner Detection Inference

Loads a trained KeypointDINO_D checkpoint and runs corner detection on a
single RGB + depth pair.  Outputs keypoints as JSON and (optionally) saves
an annotated visualisation.

CONFIGURATION
─────────────
Edit the block below, then:
    uv run python dino_d_rgb/inference.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2

from training import KeypointDINO_D, extract_keypoints

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG: Dict[str, Any] = {
    # Checkpoint written by training.py
    "checkpoint": "./checkpoints/best_model.pth",

    # Input pair
    "rgb_path":   "./data/images/sample_0000.jpg",
    "depth_path": "./data/depth/sample_0000.npy",

    # Detection threshold (0–1); lower → more detections, more noise
    "threshold": 0.3,

    # Save annotated image here (set to "" to skip)
    "vis_out": "./output/vis_sample_0000.jpg",

    # Save keypoints JSON here (set to "" to skip)
    "json_out": "./output/keypoints_sample_0000.json",

    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ═══════════════════════════════════════════════════════════════════════════════
# Preprocessing
# ═══════════════════════════════════════════════════════════════════════════════

def _preprocess(
    rgb_path: str,
    depth_path: str,
    image_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Load and preprocess one RGB+depth pair.

    Returns
    -------
    rgb_tensor   : [1, 3, H, W]  ImageNet-normalised
    depth_tensor : [1, 1, H, W]  min-max normalised to [0,1]
    orig_bgr     : [H0, W0, 3]   original image for visualisation
    """
    # RGB
    orig_bgr = cv2.imread(rgb_path)
    if orig_bgr is None:
        raise FileNotFoundError(f"RGB image not found: {rgb_path}")
    img_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    rgb_transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    rgb_tensor = rgb_transform(image=img_rgb)["image"].unsqueeze(0).to(device)

    # Depth
    depth_np = np.load(depth_path)
    if depth_np is None:
        raise FileNotFoundError(f"Depth map not found: {depth_path}")
    depth_np = depth_np.astype(np.float32)
    d_min, d_max = depth_np.min(), depth_np.max()
    depth_np = (depth_np - d_min) / (d_max - d_min + 1e-8)
    depth_np = cv2.resize(depth_np, (image_size, image_size), interpolation=cv2.INTER_LINEAR)
    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).to(device)

    return rgb_tensor, depth_tensor, orig_bgr


# ═══════════════════════════════════════════════════════════════════════════════
# Inference
# ═══════════════════════════════════════════════════════════════════════════════

def load_model(checkpoint_path: str, device: torch.device) -> Tuple[KeypointDINO_D, Dict[str, Any]]:
    """
    Reconstruct model from checkpoint.  Uses the config stored inside the
    checkpoint so the architecture always matches what was trained.
    """
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg: Dict[str, Any] = ckpt["config"]

    model = KeypointDINO_D(
        dino_channels=cfg["dino_channels"],
        depth_channels=cfg["depth_channels"],
        neck_channels=cfg["neck_channels"],
        num_decoder_blocks=cfg["num_decoder_blocks"],
        num_classes=cfg["num_classes"],
        freeze_rgb_backbone=cfg.get("freeze_rgb_backbone", True),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    epoch    = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    print(f"Loaded checkpoint: epoch={epoch}, val_loss={val_loss:.4f}")
    return model, cfg


@torch.no_grad()
def run(
    model: KeypointDINO_D,
    rgb_tensor: torch.Tensor,
    depth_tensor: torch.Tensor,
    image_size: int,
    threshold: float,
) -> Tuple[List[Tuple[float, float]], np.ndarray]:
    """
    Forward pass → heatmap → keypoints.

    Returns
    -------
    keypoints : list of (x, y) in original image_size pixel coordinates
    heatmap   : [H_map, W_map] numpy array (probabilities ∈ [0,1])
    """
    raw_out = model(rgb_tensor, depth_tensor)          # [1, 1, H_map, W_map]
    heatmap_t = torch.sigmoid(raw_out).squeeze(0)     # [1, H_map, W_map]
    keypoints = extract_keypoints(heatmap_t, threshold=threshold, image_size=image_size)
    heatmap_np = heatmap_t.squeeze(0).cpu().numpy()   # [H_map, W_map]
    return keypoints, heatmap_np


# ═══════════════════════════════════════════════════════════════════════════════
# Visualisation
# ═══════════════════════════════════════════════════════════════════════════════

def visualise(
    orig_bgr: np.ndarray,
    keypoints: List[Tuple[float, float]],
    heatmap_np: np.ndarray,
    image_size: int,
    out_path: str,
) -> None:
    """
    Draw detected corners on the original image and save alongside the heatmap.

    Layout: original image (left) | heatmap overlay (right)
    """
    orig_h, orig_w = orig_bgr.shape[:2]
    scale_x = orig_w / image_size
    scale_y = orig_h / image_size

    # Draw keypoints on a copy of the original
    vis = orig_bgr.copy()
    for (x, y) in keypoints:
        cx = int(round(x * scale_x))
        cy = int(round(y * scale_y))
        cv2.circle(vis, (cx, cy), radius=6, color=(0, 255, 0), thickness=-1)
        cv2.circle(vis, (cx, cy), radius=7, color=(0, 0, 0),   thickness=1)

    # Heatmap as colour overlay
    hmap_u8   = (np.clip(heatmap_np, 0, 1) * 255).astype(np.uint8)
    hmap_color = cv2.applyColorMap(hmap_u8, cv2.COLORMAP_JET)
    hmap_resized = cv2.resize(hmap_color, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    overlay = cv2.addWeighted(orig_bgr, 0.55, hmap_resized, 0.45, 0)

    # Draw keypoints on overlay too
    for (x, y) in keypoints:
        cx = int(round(x * scale_x))
        cy = int(round(y * scale_y))
        cv2.circle(overlay, (cx, cy), radius=6, color=(0, 255, 0), thickness=-1)
        cv2.circle(overlay, (cx, cy), radius=7, color=(0, 0, 0),   thickness=1)

    combined = np.concatenate([vis, overlay], axis=1)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(out_path, combined)
    print(f"Visualisation saved → {out_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg    = CONFIG
    device = torch.device(cfg["device"])
    print(f"Device: {device}")

    model, train_cfg = load_model(cfg["checkpoint"], device)
    image_size = train_cfg["image_size"]

    rgb_tensor, depth_tensor, orig_bgr = _preprocess(
        cfg["rgb_path"], cfg["depth_path"], image_size, device
    )

    keypoints, heatmap_np = run(
        model, rgb_tensor, depth_tensor,
        image_size=image_size,
        threshold=cfg["threshold"],
    )

    print(f"Detected {len(keypoints)} corner(s):")
    for i, (x, y) in enumerate(keypoints):
        print(f"  [{i}]  x={x:.1f}  y={y:.1f}")

    if cfg.get("json_out"):
        out = {"rgb": cfg["rgb_path"], "depth": cfg["depth_path"], "keypoints": keypoints}
        Path(cfg["json_out"]).parent.mkdir(parents=True, exist_ok=True)
        with open(cfg["json_out"], "w") as f:
            json.dump(out, f, indent=2)
        print(f"Keypoints saved → {cfg['json_out']}")

    if cfg.get("vis_out"):
        visualise(orig_bgr, keypoints, heatmap_np, image_size, cfg["vis_out"])


if __name__ == "__main__":
    main()