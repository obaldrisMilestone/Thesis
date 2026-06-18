#!/usr/bin/env python3
"""
inference.py — Run KeypointDINO on camera frames and visualise detected corners.

Loads the best checkpoint saved by training.py, runs it on every image listed
in INPUT_IMAGES, draws the predicted keypoints on the original-resolution
image, and writes results to OUTPUT_DIR.
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from training import KeypointDINO, extract_keypoints

# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════

CHECKPOINT = "./checkpoints/best_model.pth"

INPUT_IMAGES: List[str] = [
    "../dataset/Point_Detection_Tests/Camera_01_frame.jpg",
    "../dataset/Point_Detection_Tests/Camera_02_frame.jpg",
    "../dataset/Point_Detection_Tests/Camera_03_frame.jpg",
    "../dataset/Point_Detection_Tests/Camera_04_frame.jpg",
    "../dataset/Point_Detection_Tests/Camera_05_frame.jpg",
    "../dataset/Point_Detection_Tests/Camera_06_frame.jpg",
    "../dataset/Point_Detection_Tests/Camera_07_frame.jpg",
    "../dataset/Point_Detection_Tests/Camera_08_frame.jpg",
]

OUTPUT_DIR = "./results"

THRESHOLD = 0.3   # minimum heatmap confidence to keep a peak

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[KeypointDINO, dict]:
    """Load KeypointDINO from a training checkpoint; returns (model, cfg)."""
    ckpt = torch.load(checkpoint_path, map_location=device)
    cfg  = ckpt["config"]

    model = KeypointDINO(
        dino_channels      = cfg["dino_channels"],
        neck_channels      = cfg["neck_channels"],
        num_decoder_blocks = cfg["num_decoder_blocks"],
        num_classes        = cfg["num_classes"],
        freeze_backbone    = False,   # weights are already loaded; no need to re-freeze
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    print(
        f"Loaded checkpoint from epoch {ckpt['epoch']}  "
        f"(val_loss={ckpt['val_loss']:.4f})  →  {checkpoint_path}"
    )
    return model, cfg


@torch.no_grad()
def predict(
    model: KeypointDINO,
    image_path: str,
    image_size: int,
    device: torch.device,
    threshold: float,
) -> Tuple[np.ndarray, List[Tuple[float, float]]]:
    """
    Run inference on a single image.

    Returns:
        orig_bgr  : original image (unmodified, original resolution)
        keypoints : list of (x, y) in *original* image pixel coordinates
    """
    orig_bgr = cv2.imread(image_path)
    if orig_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")
    orig_h, orig_w = orig_bgr.shape[:2]

    img_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        ToTensorV2(),
    ])
    img_t = transform(image=img_rgb)["image"].unsqueeze(0).to(device)
    # img_t: [1, 3, image_size, image_size]

    raw_out = model(img_t)                           # [1, 1, H_map, W_map]
    heatmap = torch.sigmoid(raw_out).squeeze(0)     # [1, H_map, W_map]

    # extract_keypoints returns coords in image_size space → rescale to original
    kps_model_space = extract_keypoints(heatmap, threshold=threshold, image_size=image_size)

    scale_x = orig_w / image_size
    scale_y = orig_h / image_size
    keypoints = [(x * scale_x, y * scale_y) for x, y in kps_model_space]

    return orig_bgr, keypoints


def visualise(
    image_bgr: np.ndarray,
    keypoints: List[Tuple[float, float]],
) -> np.ndarray:
    """Draw detected corners on the image (returns a copy)."""
    vis = image_bgr.copy()
    for x, y in keypoints:
        cx, cy = int(round(x)), int(round(y))
        cv2.circle(vis, (cx, cy), radius=8, color=(0, 255, 0), thickness=-1)
        cv2.circle(vis, (cx, cy), radius=8, color=(0, 0, 0),   thickness=2)
    return vis


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    device = torch.device(DEVICE)
    print(f"Device: {device}")

    model, cfg = load_model(CHECKPOINT, device)

    out_dir = Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in INPUT_IMAGES:
        p = Path(img_path)
        if not p.exists():
            print(f"  [skip] not found: {img_path}")
            continue

        orig_bgr, keypoints = predict(
            model,
            img_path,
            image_size=cfg["image_size"],
            device=device,
            threshold=THRESHOLD,
        )

        vis = visualise(orig_bgr, keypoints)

        stem      = p.stem
        out_path  = out_dir / f"{stem}_corners.jpg"
        cv2.imwrite(str(out_path), vis)

        print(f"  {stem}: {len(keypoints)} corner(s) detected  →  {out_path}")


if __name__ == "__main__":
    main()
