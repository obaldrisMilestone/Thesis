#!/usr/bin/env python3
"""
training.py — RGB-D Room Corner Detection via DINOv2 + ResNet18 Late Fusion

Dual-stream architecture for ablation study. Detects room corners (floor-wall
intersections) from RGB + Depth pairs using:
  RGB stream  : Frozen DINOv2-S (semantic + geometric features)
  Depth stream: Trainable ResNet18 (modified for 1-ch input)
  Fusion      : Late concatenation → 1×1 Conv neck → decoder → heatmap head

Script layout
─────────────
  1. Config
  2. Model  (DepthResNet18 / KeypointDINO_D: backbone A+B / neck C / decoder D / head E)
  3. Mock data generator  (RGB images + synthetic depth maps)
  4. Dataset  (RoomCornerRGBDDataset + generate_gaussian_heatmap)
  5. Loss  (HeatmapFocalLoss — CenterNet style)
  6. Inference  (heatmap → (x,y) via NMS)
  7. Training utilities  (train_one_epoch / evaluate)
  8. main()
"""

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Config
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG: Dict[str, Any] = {
    # ── Image / model ──────────────────────────────────────────────────────────
    "image_size": 224,           # H = W; must be divisible by patch_size (14)
    "patch_size": 14,            # DINOv2-S patch size
    "dino_channels": 384,        # DINOv2-S embedding dim
    "depth_channels": 512,       # ResNet18 layer4 output channels
    "neck_channels": 256,        # bottleneck width after fusion 1×1 conv
    "num_decoder_blocks": 2,     # upscale factor = 2^N → 64×64 heatmap for 224px input
    "num_classes": 1,            # 1 = "any room corner"
    "freeze_rgb_backbone": True, # freeze DINOv2 entirely
    # ── Heatmap ────────────────────────────────────────────────────────────────
    "sigma": 2.0,                # Gaussian spread in heatmap-pixel units
    # ── Training ───────────────────────────────────────────────────────────────
    "batch_size": 8,
    "num_epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "val_split": 0.2,
    "num_workers": 4,
    "log_interval": 10,          # print batch loss every N steps
    # ── Paths ──────────────────────────────────────────────────────────────────
    "data_dir": "./data",
    "checkpoint_dir": "./checkpoints",
    # ── Runtime ────────────────────────────────────────────────────────────────
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_mock_samples": 300,
}

# ═══════════════════════════════════════════════════════════════════════════════
# 2. Model Architecture
# ═══════════════════════════════════════════════════════════════════════════════

class UpsampleBlock(nn.Module):
    """Bilinear 2× upsample → Conv 3×3 → BN → ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C_in, H, W]
        x = F.interpolate(x, scale_factor=2.0, mode="bilinear", align_corners=False)
        # x: [B, C_in, 2H, 2W]
        x = F.relu(self.bn(self.conv(x)), inplace=True)
        # x: [B, C_out, 2H, 2W]
        return x


class DepthResNet18(nn.Module):
    """
    Modified ResNet18 for single-channel depth input.

    Modifications:
      1. conv1 adapted from 3→1 input channel; new weights = sum of original 3-ch
         filters (preserves pre-trained edge-detection structure).
      2. fc and avgpool removed.
      3. Forward accepts a target spatial size (ph, pw) and uses
         F.adaptive_avg_pool2d to align with the DINOv2 patch grid.

    Output: [B, 512, ph, pw]
    """

    def __init__(self) -> None:
        super().__init__()
        resnet = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)

        # Modify conv1: 3-channel → 1-channel
        old_w = resnet.conv1.weight.data                              # [64, 3, 7, 7]
        new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        new_conv1.weight.data = old_w.sum(dim=1, keepdim=True)       # [64, 1, 7, 7]
        resnet.conv1 = new_conv1

        # Stem: conv1 → bn1 → relu → maxpool
        self.stem   = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1   # [B, 64,  H/4,  W/4]
        self.layer2 = resnet.layer2   # [B, 128, H/8,  W/8]
        self.layer3 = resnet.layer3   # [B, 256, H/16, W/16]
        self.layer4 = resnet.layer4   # [B, 512, H/32, W/32]
        # fc and avgpool intentionally omitted

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        # x: [B, 1, H, W]
        x = self.stem(x)     # [B, 64,  H/4,  W/4]
        x = self.layer1(x)   # [B, 64,  H/4,  W/4]
        x = self.layer2(x)   # [B, 128, H/8,  W/8]
        x = self.layer3(x)   # [B, 256, H/16, W/16]
        x = self.layer4(x)   # [B, 512, H/32, W/32]
        # Align spatial dims to DINOv2 patch grid (e.g. 7×7 → 16×16 for 224px input)
        x = F.adaptive_avg_pool2d(x, target_hw)
        # x: [B, 512, ph, pw]
        return x


class KeypointDINO_D(nn.Module):
    """
    RGB-D room corner detector — late-fusion dual-stream.

    Forward pass (224×224 input, 2 decoder blocks):
      RGB stream   : [B,  3, 224, 224] → [B, 384, 16, 16]   (frozen DINOv2-S)
      Depth stream : [B,  1, 224, 224] → [B, 512, 16, 16]   (trainable ResNet18)
      Concatenation:                    → [B, 896, 16, 16]
      Neck (1×1)   :                    → [B, 256, 16, 16]
      Decoder      : 2× UpsampleBlock  → [B,  64, 64, 64]
      Head (1×1)   :                    → [B,   1, 64, 64]   (raw logits)
    """

    def __init__(
        self,
        dino_channels: int = 384,
        depth_channels: int = 512,
        neck_channels: int = 256,
        num_decoder_blocks: int = 2,
        num_classes: int = 1,
        freeze_rgb_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = 14

        # ── A. RGB Stream — frozen DINOv2-S ───────────────────────────────────
        self.rgb_backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        if freeze_rgb_backbone:
            for param in self.rgb_backbone.parameters():
                param.requires_grad = False

        # ── B. Depth Stream — trainable ResNet18 ──────────────────────────────
        self.depth_encoder = DepthResNet18()

        # ── C. Neck — channel reduction after late fusion ─────────────────────
        fused_ch = dino_channels + depth_channels   # 384 + 512 = 896
        self.neck = nn.Sequential(
            nn.Conv2d(fused_ch, neck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(neck_channels),
            nn.ReLU(inplace=True),
        )

        # ── D. Decoder — N × UpsampleBlock ───────────────────────────────────
        layers: List[nn.Module] = []
        in_ch = neck_channels
        for i in range(num_decoder_blocks):
            out_ch = max(neck_channels // (2 ** (i + 1)), 64)
            layers.append(UpsampleBlock(in_ch, out_ch))
            in_ch = out_ch
        self.decoder = nn.Sequential(*layers)
        self._dec_out_ch = in_ch

        # ── E. Head — 1×1 Conv to logits ─────────────────────────────────────
        self.head = nn.Conv2d(self._dec_out_ch, num_classes, kernel_size=1)

    def forward(self, rgb: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        B, _C, H, W = rgb.shape
        ph = H // self.patch_size   # DINOv2 spatial grid height  e.g. 16
        pw = W // self.patch_size   # DINOv2 spatial grid width

        # ── A. RGB features (frozen) ──────────────────────────────────────────
        features     = self.rgb_backbone.forward_features(rgb)
        # features["x_norm_patchtokens"]: [B, ph*pw, dino_channels]
        patch_tokens = features["x_norm_patchtokens"]
        rgb_feat     = patch_tokens.reshape(B, ph, pw, -1).permute(0, 3, 1, 2).contiguous()
        # rgb_feat: [B, dino_channels, ph, pw]

        # ── B. Depth features (trainable) ─────────────────────────────────────
        depth_feat = self.depth_encoder(depth, target_hw=(ph, pw))
        # depth_feat: [B, depth_channels, ph, pw]

        # ── C. Late fusion + Neck ─────────────────────────────────────────────
        fused = torch.cat([rgb_feat, depth_feat], dim=1)
        # fused: [B, dino_channels + depth_channels, ph, pw]
        feat = self.neck(fused)
        # feat: [B, neck_channels, ph, pw]

        # ── D. Decoder ────────────────────────────────────────────────────────
        feat = self.decoder(feat)
        # feat: [B, dec_out_ch, ph·2^N, pw·2^N]

        # ── E. Head ───────────────────────────────────────────────────────────
        heatmap = self.head(feat)
        # heatmap: [B, num_classes, H_map, W_map]  (raw logits)
        return heatmap


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Mock Data Generator
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_mock_data(data_dir: str, num_samples: int, image_size: int) -> str:
    """
    Generate synthetic indoor RGB images and companion depth maps with
    floor-wall intersection keypoints.

    Directory layout:
      <data_dir>/images/          ← RGB JPEGs
      <data_dir>/depth/           ← float32 .npy depth maps  (values ∈ [0,1])
      <data_dir>/annotations/keypoints.json

    Returns path to the annotation JSON.
    """
    img_dir   = Path(data_dir) / "images"
    depth_dir = Path(data_dir) / "depth"
    ann_dir   = Path(data_dir) / "annotations"
    for d in (img_dir, depth_dir, ann_dir):
        d.mkdir(parents=True, exist_ok=True)

    ann_file = ann_dir / "keypoints.json"
    if ann_file.exists():
        print(f"Mock data already present at {data_dir} — skipping generation.")
        return str(ann_file)

    annotations: Dict[str, List[List[float]]] = {}

    for i in range(num_samples):
        H = W = image_size

        # ── RGB ───────────────────────────────────────────────────────────────
        wall_val = random.randint(170, 220)
        img = np.full((H, W, 3), wall_val, dtype=np.uint8)

        floor_y = int(H * random.uniform(0.45, 0.65))
        floor_color = (
            random.randint(80, 150),
            random.randint(70, 130),
            random.randint(60, 110),
        )
        img[floor_y:, :] = floor_color

        num_walls = random.randint(1, 3)
        wx_pool = list(range(20, W - 20))
        random.shuffle(wx_pool)
        wall_xs = sorted(wx_pool[:num_walls])

        keypoints: List[List[float]] = []
        for wx in wall_xs:
            shade = max(0, wall_val - random.randint(30, 70))
            cv2.line(img, (wx, 0), (wx, floor_y), (shade, shade, shade), 2)
            keypoints.append([float(wx) + random.uniform(-1.5, 1.5),
                               float(floor_y) + random.uniform(-1.5, 1.5)])

        if random.random() > 0.4:
            keypoints.append([random.uniform(3, 18),
                               float(floor_y) + random.uniform(-2.0, 2.0)])
        if random.random() > 0.4:
            keypoints.append([float(W) - random.uniform(3, 18),
                               float(floor_y) + random.uniform(-2.0, 2.0)])

        # ── Depth ─────────────────────────────────────────────────────────────
        # Walls/ceiling: distant → low depth
        # Floor: closer → depth increases toward camera (bottom of frame)
        depth = np.zeros((H, W), dtype=np.float32)
        noise = np.random.uniform(-0.04, 0.04, (H, W)).astype(np.float32)

        wall_base = np.random.uniform(0.05, 0.35, (floor_y, W)).astype(np.float32)
        depth[:floor_y, :] = np.clip(wall_base + noise[:floor_y, :], 0.0, 1.0)

        for y in range(floor_y, H):
            t = (y - floor_y) / max(1, H - floor_y)   # 0 at horizon, 1 at bottom
            row = np.random.uniform(0.35 + 0.4 * t, 0.55 + 0.4 * t, W).astype(np.float32)
            depth[y, :] = np.clip(row + noise[y, :], 0.0, 1.0)

        fname = f"sample_{i:04d}"
        cv2.imwrite(str(img_dir / f"{fname}.jpg"), img)
        np.save(str(depth_dir / f"{fname}.npy"), depth)
        annotations[f"{fname}.jpg"] = keypoints

    with open(ann_file, "w") as f:
        json.dump(annotations, f)

    print(f"Generated {num_samples} mock RGB-D samples → {data_dir}")
    return str(ann_file)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Dataset
# ═══════════════════════════════════════════════════════════════════════════════

def generate_gaussian_heatmap(
    heatmap_hw: Tuple[int, int],
    keypoints: List[List[float]],
    sigma: float,
) -> np.ndarray:
    """
    Render a multi-keypoint Gaussian heatmap.
    Returns float32 [H_map, W_map], values ∈ [0,1]; peak = 1.0 at each keypoint.
    """
    H, W = heatmap_hw
    heatmap = np.zeros((H, W), dtype=np.float32)
    if not keypoints:
        return heatmap

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    two_sig_sq = 2.0 * sigma ** 2

    for kp in keypoints:
        kx, ky = float(kp[0]), float(kp[1])
        gaussian = np.exp(-((xx - kx) ** 2 + (yy - ky) ** 2) / two_sig_sq)
        np.maximum(heatmap, gaussian, out=heatmap)   # keep element-wise max

    return heatmap   # [H_map, W_map]


class RoomCornerRGBDDataset(Dataset):
    """
    Indoor room corner dataset (RGB + Depth pairs).

    Augmentation strategy:
      - Geometric transforms (HorizontalFlip, RandomResizedCrop) are recorded via
        A.ReplayCompose and replayed identically on the depth map so both inputs
        stay spatially aligned.
      - ColorJitter is applied to RGB only (depth is invariant to illumination).
      - Rotations are strictly forbidden.

    Returns per item:
      rgb    : Tensor [3, H, W]         (ImageNet-normalised)
      depth  : Tensor [1, H, W]         (per-sample min-max normalised to [0,1])
      heatmap: Tensor [1, H_map, W_map] (Gaussian, float32 ∈ [0,1])
    """

    def __init__(
        self,
        samples: List[Tuple[str, List[List[float]]]],
        data_dir: str,
        image_size: int,
        heatmap_size: int,
        sigma: float,
        augment: bool,
    ) -> None:
        self.samples      = samples
        self.img_dir      = Path(data_dir) / "images"
        self.depth_dir    = Path(data_dir) / "depth"
        self.image_size   = image_size
        self.heatmap_size = heatmap_size
        self.sigma        = sigma
        self.kp_scale     = heatmap_size / image_size

        # Geometric transforms recorded for replay on depth
        if augment:
            self.geo_transform = A.ReplayCompose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomResizedCrop(
                        height=image_size,
                        width=image_size,
                        scale=(0.7, 1.0),
                        ratio=(0.9, 1.1),
                    ),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
            )
            # ColorJitter applied to RGB only after geometry
            self.rgb_color_aug = A.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=0.8
            )
        else:
            self.geo_transform = A.ReplayCompose(
                [
                    A.Resize(height=image_size, width=image_size),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
            )
            self.rgb_color_aug = None

        self.rgb_normalize = A.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        img_name, raw_kps = self.samples[idx]
        stem = Path(img_name).stem   # e.g. "sample_0000"

        # ── Load RGB ──────────────────────────────────────────────────────────
        img_bgr = cv2.imread(str(self.img_dir / img_name))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)   # [H, W, 3] uint8

        # ── Load depth → [0,1] float32 ───────────────────────────────────────
        depth_np = np.load(str(self.depth_dir / f"{stem}.npy")).astype(np.float32)
        d_min, d_max = depth_np.min(), depth_np.max()
        depth_np = (depth_np - d_min) / (d_max - d_min + 1e-8)   # [H, W], ∈ [0,1]

        # ── Geometric augmentation on RGB + keypoints (record replay) ─────────
        albu_kps = [(float(kp[0]), float(kp[1])) for kp in raw_kps]
        rgb_result = self.geo_transform(image=img_rgb, keypoints=albu_kps)

        img_geo = rgb_result["image"]           # [H, W, 3] uint8, geometrically augmented
        aug_kps = rgb_result["keypoints"]       # list of (x, y) after geometry

        # ── Replay exact same geometry on depth ───────────────────────────────
        # Convert to uint8 for albumentations replay (3-ch format required for image type)
        depth_u8  = (depth_np * 255.0).clip(0, 255).astype(np.uint8)
        depth_rgb = np.stack([depth_u8, depth_u8, depth_u8], axis=2)   # [H, W, 3]
        depth_replayed = A.ReplayCompose.replay(
            rgb_result["replay"], image=depth_rgb
        )
        depth_geo = depth_replayed["image"][:, :, 0].astype(np.float32) / 255.0
        # depth_geo: [H, W], ∈ [0,1], geometrically aligned with img_geo

        # ── ColorJitter on RGB only ───────────────────────────────────────────
        if self.rgb_color_aug is not None:
            img_geo = self.rgb_color_aug(image=img_geo)["image"]

        # ── Normalise RGB (ImageNet stats) → tensor ───────────────────────────
        img_norm   = self.rgb_normalize(image=img_geo)["image"]          # [H, W, 3] float
        rgb_tensor = torch.from_numpy(img_norm.transpose(2, 0, 1))       # [3, H, W]

        # ── Depth → tensor ────────────────────────────────────────────────────
        depth_tensor = torch.from_numpy(depth_geo).unsqueeze(0)          # [1, H, W]

        # ── Scale keypoints → heatmap space and build GT heatmap ──────────────
        hmap_kps = [[kp[0] * self.kp_scale, kp[1] * self.kp_scale] for kp in aug_kps]
        heatmap_np = generate_gaussian_heatmap(
            (self.heatmap_size, self.heatmap_size), hmap_kps, sigma=self.sigma
        )
        heatmap_tensor = torch.from_numpy(heatmap_np).unsqueeze(0)       # [1, H_map, W_map]

        return rgb_tensor, depth_tensor, heatmap_tensor


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Loss Function — CenterNet-style Penalty-Reduced Focal Loss
# ═══════════════════════════════════════════════════════════════════════════════

class HeatmapFocalLoss(nn.Module):
    """
    Penalty-reduced focal loss (CornerNet / CenterNet variant).

    Positive samples  (gt == 1.0): standard focal loss term.
    Negative samples  (gt  < 1.0): down-weighted by (1 - gt)^beta so pixels
                                    near a peak incur far less penalty.

    Args:
        alpha : exponent focusing loss on hard examples (default 2)
        beta  : exponent for near-peak penalty reduction (default 4)
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta  = beta

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # pred: [B, C, H_map, W_map]  — raw logits from the head
        # gt  : [B, C, H_map, W_map]  — Gaussian heatmap, values ∈ [0, 1]

        # Sigmoid + clamp to avoid log(0)
        p = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1.0 - 1e-4)
        # p: [B, C, H_map, W_map]

        pos_mask    = (gt == 1.0).float()               # [B, C, H_map, W_map]
        neg_mask    = 1.0 - pos_mask                    # [B, C, H_map, W_map]
        neg_weights = torch.pow(1.0 - gt, self.beta)   # [B, C, H_map, W_map]

        # Positive focal loss — high at low-confidence peaks
        pos_loss = pos_mask * torch.pow(1.0 - p, self.alpha) * torch.log(p)
        # [B, C, H_map, W_map]

        # Negative focal loss — penalty reduced near Gaussian peaks
        neg_loss = neg_mask * neg_weights * torch.pow(p, self.alpha) * torch.log(1.0 - p)
        # [B, C, H_map, W_map]

        num_pos = pos_mask.sum().clamp(min=1.0)
        loss    = -(pos_loss.sum() + neg_loss.sum()) / num_pos
        return loss


# ═══════════════════════════════════════════════════════════════════════════════
# 6. Inference — Heatmap → (x, y) Keypoint Coordinates
# ═══════════════════════════════════════════════════════════════════════════════

def extract_keypoints(
    heatmap: torch.Tensor,
    threshold: float = 0.3,
    image_size: int = 224,
) -> List[Tuple[float, float]]:
    """
    Convert a predicted heatmap to image-space (x, y) coordinates.

    NMS via 2D MaxPool (kernel 3×3, stride 1, pad 1): a pixel survives iff it
    equals the local maximum AND is above `threshold`.

    Args:
        heatmap    : [1, H_map, W_map]  — after sigmoid, values ∈ [0, 1]
        threshold  : minimum confidence to keep a peak
        image_size : original image side length (assumes square input)

    Returns:
        List of (x, y) tuples in original image pixel coordinates.
    """
    # heatmap: [1, H_map, W_map]
    H_map, W_map = heatmap.shape[1], heatmap.shape[2]

    pooled = F.max_pool2d(
        heatmap.unsqueeze(0),   # [1, 1, H_map, W_map]
        kernel_size=3,
        stride=1,
        padding=1,
    ).squeeze(0)                # [1, H_map, W_map]

    keep = (heatmap == pooled) & (heatmap >= threshold)   # [1, H_map, W_map]

    scale_x = image_size / W_map
    scale_y = image_size / H_map

    indices   = keep[0].nonzero(as_tuple=False)           # [N, 2] — (row, col)
    keypoints = [
        (float(col) * scale_x, float(row) * scale_y)
        for row, col in indices.tolist()
    ]
    return keypoints


def run_inference(
    model: nn.Module,
    rgb_path: str,
    depth_path: str,
    cfg: Dict[str, Any],
    threshold: float = 0.3,
) -> List[Tuple[float, float]]:
    """
    End-to-end inference on a single RGB + depth pair.
    Returns (x, y) keypoints in original image pixel coordinates.

    Args:
        rgb_path   : path to RGB image (JPEG/PNG)
        depth_path : path to depth map (.npy, float32 [H, W])
    """
    device = torch.device(cfg["device"])
    model.eval()

    img_bgr = cv2.imread(rgb_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    rgb_transform = A.Compose([
        A.Resize(height=cfg["image_size"], width=cfg["image_size"]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    rgb_tensor = rgb_transform(image=img_rgb)["image"].unsqueeze(0).to(device)
    # rgb_tensor: [1, 3, H, W]

    depth_np = np.load(depth_path).astype(np.float32)
    d_min, d_max = depth_np.min(), depth_np.max()
    depth_np = (depth_np - d_min) / (d_max - d_min + 1e-8)
    depth_np = cv2.resize(depth_np, (cfg["image_size"], cfg["image_size"]),
                          interpolation=cv2.INTER_LINEAR)
    depth_tensor = torch.from_numpy(depth_np).unsqueeze(0).unsqueeze(0).to(device)
    # depth_tensor: [1, 1, H, W]

    with torch.no_grad():
        raw_out = model(rgb_tensor, depth_tensor)               # [1, 1, H_map, W_map]
        heatmap = torch.sigmoid(raw_out).squeeze(0)             # [1, H_map, W_map]

    return extract_keypoints(heatmap, threshold=threshold, image_size=cfg["image_size"])


# ═══════════════════════════════════════════════════════════════════════════════
# 7. Training Utilities
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    log_interval: int,
) -> float:
    model.train()
    running = 0.0

    for step, (rgb, depth, heatmaps) in enumerate(loader, start=1):
        rgb      = rgb.to(device, non_blocking=True)        # [B, 3, H, W]
        depth    = depth.to(device, non_blocking=True)      # [B, 1, H, W]
        heatmaps = heatmaps.to(device, non_blocking=True)   # [B, 1, H_map, W_map]

        optimizer.zero_grad(set_to_none=True)
        pred = model(rgb, depth)                            # [B, 1, H_map, W_map]
        loss = criterion(pred, heatmaps)
        loss.backward()
        optimizer.step()

        running += loss.item()
        if step % log_interval == 0:
            print(f"  Epoch {epoch} | Step {step:>4}/{len(loader)} | Loss {running / step:.4f}")

    return running / len(loader)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.eval()
    total = 0.0
    for rgb, depth, heatmaps in loader:
        rgb      = rgb.to(device, non_blocking=True)
        depth    = depth.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)
        pred   = model(rgb, depth)                          # [B, 1, H_map, W_map]
        total += criterion(pred, heatmaps).item()
    return total / len(loader)


# ═══════════════════════════════════════════════════════════════════════════════
# 8. main()
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    cfg    = CONFIG
    device = torch.device(cfg["device"])
    print(f"Device: {device}")

    Path(cfg["checkpoint_dir"]).mkdir(parents=True, exist_ok=True)

    # ── Mock data ─────────────────────────────────────────────────────────────
    ann_file = _generate_mock_data(
        cfg["data_dir"],
        num_samples=cfg["num_mock_samples"],
        image_size=cfg["image_size"],
    )

    with open(ann_file) as f:
        all_ann = json.load(f)

    samples = list(all_ann.items())
    random.shuffle(samples)

    n_val   = int(len(samples) * cfg["val_split"])
    n_train = len(samples) - n_val
    train_samples, val_samples = samples[:n_train], samples[n_train:]

    # Heatmap spatial size: patch_grid × 2^num_decoder_blocks
    # e.g. 224px → grid 16×16 → 2 blocks → heatmap 64×64
    patch_grid   = cfg["image_size"] // cfg["patch_size"]
    heatmap_size = patch_grid * (2 ** cfg["num_decoder_blocks"])

    print(
        f"Image {cfg['image_size']}px  |  "
        f"Patch grid {patch_grid}×{patch_grid}  |  "
        f"Heatmap {heatmap_size}×{heatmap_size}  |  "
        f"Train {n_train}  Val {n_val}"
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = RoomCornerRGBDDataset(
        samples=train_samples,
        data_dir=cfg["data_dir"],
        image_size=cfg["image_size"],
        heatmap_size=heatmap_size,
        sigma=cfg["sigma"],
        augment=True,
    )
    val_ds = RoomCornerRGBDDataset(
        samples=val_samples,
        data_dir=cfg["data_dir"],
        image_size=cfg["image_size"],
        heatmap_size=heatmap_size,
        sigma=cfg["sigma"],
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        pin_memory=(device.type == "cuda"),
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = KeypointDINO_D(
        dino_channels=cfg["dino_channels"],
        depth_channels=cfg["depth_channels"],
        neck_channels=cfg["neck_channels"],
        num_decoder_blocks=cfg["num_decoder_blocks"],
        num_classes=cfg["num_classes"],
        freeze_rgb_backbone=cfg["freeze_rgb_backbone"],
    ).to(device)

    # Only depth stream + neck + decoder + head are trainable
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Trainable tensors: {len(trainable)}  |  "
        f"Parameters: {sum(p.numel() for p in trainable):,}"
    )

    # ── Optimizer (depth stream + neck + decoder + head only) ─────────────────
    optimizer = torch.optim.AdamW(
        trainable,
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg["num_epochs"],
        eta_min=1e-6,
    )
    criterion = HeatmapFocalLoss(alpha=2.0, beta=4.0)

    # ── Training loop ─────────────────────────────────────────────────────────
    best_val  = float("inf")
    ckpt_path = Path(cfg["checkpoint_dir"]) / "best_model.pth"

    for epoch in range(1, cfg["num_epochs"] + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion,
            device, epoch, cfg["log_interval"],
        )
        val_loss = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        lr_now = scheduler.get_last_lr()[0]
        print(
            f"Epoch [{epoch:>3}/{cfg['num_epochs']}]  "
            f"Train {train_loss:.4f}  Val {val_loss:.4f}  LR {lr_now:.2e}"
        )

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "config": cfg,
                },
                ckpt_path,
            )
            print(f"  >> Best model saved  (val_loss={val_loss:.4f})  →  {ckpt_path}")


if __name__ == "__main__":
    main()