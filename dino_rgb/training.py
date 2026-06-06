#!/usr/bin/env python3
"""
training.py — RGB-Only Room Corner Detection via DINOv2 + Heatmap Regression

Baseline architecture for ablation study. Detects room corners (floor-wall
intersections) from single RGB images using a frozen DINOv2-S backbone +
lightweight neck + decoder + focal-loss heatmap head.

Script layout
─────────────
  1. Config
  2. Model  (KeypointDINO: backbone / neck / decoder / head)
  3. Mock data generator
  4. Dataset  (RoomCornerDataset + generate_gaussian_heatmap)
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
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

# ═══════════════════════════════════════════════════════════════════════════════
# 1. Config
# ═══════════════════════════════════════════════════════════════════════════════

CONFIG: Dict[str, Any] = {
    # ── Image / model ──────────────────────────────────────────────────────────
    "image_size": 224,          # H = W; must be divisible by patch_size (14)
    "patch_size": 14,           # DINOv2-S patch size
    "dino_channels": 384,       # DINOv2-S embedding dim
    "neck_channels": 256,       # bottleneck channel width after 1×1 conv
    "num_decoder_blocks": 2,    # upscale factor = 2^N  →  64×64 heatmap for 224px input
    "num_classes": 1,           # 1 = "any room corner"
    "freeze_backbone": True,    # freeze DINOv2 entirely
    # ── Heatmap ────────────────────────────────────────────────────────────────
    "sigma": 2.0,               # Gaussian spread in heatmap-pixel units
    # ── Training ───────────────────────────────────────────────────────────────
    "batch_size": 8,
    "num_epochs": 50,
    "lr": 1e-4,
    "weight_decay": 1e-4,
    "val_split": 0.2,
    "num_workers": 4,
    "log_interval": 10,         # print batch loss every N steps
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


class KeypointDINO(nn.Module):
    """
    RGB-only room corner detector.

    Forward pass dimensions (default config, 224×224 input):
      Backbone  : [B, 3, 224, 224] → patch tokens [B, 256, 384]
      Reshape   : [B, 384, 16, 16]
      Neck      : [B, 384, 16, 16] → [B, 256, 16, 16]
      Decoder   : [B, 256, 16, 16] → [B, 128, 32, 32] → [B, 64, 64, 64]
      Head      : [B, 64, 64, 64]  → [B, 1,  64, 64]   (raw logits)
    """

    def __init__(
        self,
        dino_channels: int = 384,
        neck_channels: int = 256,
        num_decoder_blocks: int = 2,
        num_classes: int = 1,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.patch_size = 14

        # ── A. RGB Stream — frozen DINOv2-S ───────────────────────────────────
        self.backbone = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # ── B. Neck — 1×1 Conv + BN + ReLU ───────────────────────────────────
        self.neck = nn.Sequential(
            nn.Conv2d(dino_channels, neck_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(neck_channels),
            nn.ReLU(inplace=True),
        )

        # ── C. Decoder — N × UpsampleBlock ───────────────────────────────────
        layers: List[nn.Module] = []
        in_ch = neck_channels
        for i in range(num_decoder_blocks):
            out_ch = max(neck_channels // (2 ** (i + 1)), 64)
            layers.append(UpsampleBlock(in_ch, out_ch))
            in_ch = out_ch
        self.decoder = nn.Sequential(*layers)
        self._dec_out_ch = in_ch

        # ── D. Head — 1×1 Conv to logits ─────────────────────────────────────
        self.head = nn.Conv2d(self._dec_out_ch, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, _C, H, W = x.shape
        ph = H // self.patch_size   # spatial grid height  e.g. 224//14 = 16
        pw = W // self.patch_size   # spatial grid width

        # ── A. Extract patch tokens ───────────────────────────────────────────
        features = self.backbone.forward_features(x)
        # features["x_norm_patchtokens"]: [B, ph*pw, dino_channels]
        patch_tokens = features["x_norm_patchtokens"]

        # Reshape 1D token sequence → 2D spatial grid
        feat = patch_tokens.reshape(B, ph, pw, -1).permute(0, 3, 1, 2).contiguous()
        # feat: [B, dino_channels, ph, pw]

        # ── B. Neck ───────────────────────────────────────────────────────────
        feat = self.neck(feat)
        # feat: [B, neck_channels, ph, pw]

        # ── C. Decoder ────────────────────────────────────────────────────────
        feat = self.decoder(feat)
        # feat: [B, dec_out_ch, ph·2^N, pw·2^N]

        # ── D. Head ───────────────────────────────────────────────────────────
        heatmap = self.head(feat)
        # heatmap: [B, num_classes, H_map, W_map]  (raw logits)
        return heatmap


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Mock Data Generator
# ═══════════════════════════════════════════════════════════════════════════════

def _generate_mock_data(data_dir: str, num_samples: int, image_size: int) -> str:
    """
    Synthesise indoor RGB scenes with floor-wall intersection keypoints.
    Saves images to  <data_dir>/images/
    Saves annotations to <data_dir>/annotations/keypoints.json
    Returns path to the annotation JSON.
    """
    img_dir = Path(data_dir) / "images"
    ann_dir = Path(data_dir) / "annotations"
    img_dir.mkdir(parents=True, exist_ok=True)
    ann_dir.mkdir(parents=True, exist_ok=True)

    ann_file = ann_dir / "keypoints.json"
    if ann_file.exists():
        print(f"Mock data already present at {data_dir} — skipping generation.")
        return str(ann_file)

    annotations: Dict[str, List[List[float]]] = {}

    for i in range(num_samples):
        # Base wall colour
        wall_val = random.randint(170, 220)
        img = np.full((image_size, image_size, 3), wall_val, dtype=np.uint8)

        # Floor horizon
        floor_y = int(image_size * random.uniform(0.45, 0.65))
        floor_color = (
            random.randint(80, 150),
            random.randint(70, 130),
            random.randint(60, 110),
        )
        img[floor_y:, :] = floor_color

        # Vertical wall dividers
        num_walls = random.randint(1, 3)
        wx_candidates = list(range(20, image_size - 20))
        random.shuffle(wx_candidates)
        wall_xs = sorted(wx_candidates[:num_walls])

        keypoints: List[List[float]] = []
        for wx in wall_xs:
            shade = max(0, wall_val - random.randint(30, 70))
            cv2.line(img, (wx, 0), (wx, floor_y), (shade, shade, shade), 2)
            kx = float(wx) + random.uniform(-1.5, 1.5)
            ky = float(floor_y) + random.uniform(-1.5, 1.5)
            keypoints.append([kx, ky])

        # Left / right edge corners
        if random.random() > 0.4:
            keypoints.append([random.uniform(3, 18), float(floor_y) + random.uniform(-2, 2)])
        if random.random() > 0.4:
            keypoints.append([float(image_size) - random.uniform(3, 18), float(floor_y) + random.uniform(-2, 2)])

        fname = f"sample_{i:04d}.jpg"
        cv2.imwrite(str(img_dir / fname), img)
        annotations[fname] = keypoints

    with open(ann_file, "w") as f:
        json.dump(annotations, f)

    print(f"Generated {num_samples} mock samples → {data_dir}")
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

    Args:
        heatmap_hw : (H_map, W_map) target heatmap resolution
        keypoints  : list of [x, y] already scaled to heatmap coordinate space
        sigma      : Gaussian standard deviation in heatmap pixels

    Returns:
        float32 array [H_map, W_map], values ∈ [0, 1]; peak = 1.0 at each keypoint
    """
    H, W = heatmap_hw
    heatmap = np.zeros((H, W), dtype=np.float32)

    if not keypoints:
        return heatmap

    # Pre-build coordinate grids once for efficiency
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)   # both [H, W]
    two_sig_sq = 2.0 * sigma ** 2

    for kp in keypoints:
        kx, ky = float(kp[0]), float(kp[1])
        gaussian = np.exp(-((xx - kx) ** 2 + (yy - ky) ** 2) / two_sig_sq)
        # gaussian: [H, W], peak 1.0 at (kx, ky)
        np.maximum(heatmap, gaussian, out=heatmap)    # element-wise max keeps all peaks

    return heatmap  # [H_map, W_map]


class RoomCornerDataset(Dataset):
    """
    Custom dataset for indoor room corner detection.

    Returns per item:
      image  : Tensor [3, H, W]           (ImageNet-normalised)
      heatmap: Tensor [1, H_map, W_map]   (Gaussian, float32 ∈ [0,1])
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
        self.image_size   = image_size
        self.heatmap_size = heatmap_size
        self.sigma        = sigma
        # coordinate scale: image-space → heatmap-space
        self.kp_scale     = heatmap_size / image_size

        if augment:
            # NO rotations (STRICTLY FORBIDDEN per spec)
            self.transform = A.Compose(
                [
                    A.HorizontalFlip(p=0.5),
                    A.RandomResizedCrop(
                        height=image_size,
                        width=image_size,
                        scale=(0.7, 1.0),
                        ratio=(0.9, 1.1),
                    ),
                    A.ColorJitter(
                        brightness=0.3,
                        contrast=0.3,
                        saturation=0.3,
                        hue=0.1,
                        p=0.8,
                    ),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
            )
        else:
            self.transform = A.Compose(
                [
                    A.Resize(height=image_size, width=image_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2(),
                ],
                keypoint_params=A.KeypointParams(format="xy", remove_invisible=True),
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_name, raw_kps = self.samples[idx]

        img_bgr = cv2.imread(str(self.img_dir / img_name))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        # img_rgb: [H, W, 3]

        albu_kps = [(float(kp[0]), float(kp[1])) for kp in raw_kps]
        result = self.transform(image=img_rgb, keypoints=albu_kps)

        image_tensor: torch.Tensor = result["image"]          # [3, H, W]
        aug_kps: List[Tuple[float, float]] = result["keypoints"]

        # Scale image-space keypoints → heatmap-space
        hmap_kps = [[kp[0] * self.kp_scale, kp[1] * self.kp_scale] for kp in aug_kps]

        heatmap_np = generate_gaussian_heatmap(
            (self.heatmap_size, self.heatmap_size),
            hmap_kps,
            sigma=self.sigma,
        )
        heatmap_tensor = torch.from_numpy(heatmap_np).unsqueeze(0)  # [1, H_map, W_map]

        return image_tensor, heatmap_tensor


# ═══════════════════════════════════════════════════════════════════════════════
# 5. Loss Function — CenterNet-style Penalty-Reduced Focal Loss
# ═══════════════════════════════════════════════════════════════════════════════

class HeatmapFocalLoss(nn.Module):
    """
    Penalty-reduced focal loss (CornerNet / CenterNet variant).

    Positive samples  (gt == 1.0): standard focal loss term.
    Negative samples  (gt  < 1.0): down-weighted by (1 - gt)^beta so pixels
                                    near a peak are penalised far less.

    Args:
        alpha : exponent controlling focus on hard examples (default 2)
        beta  : exponent controlling near-peak penalty reduction (default 4)
    """

    def __init__(self, alpha: float = 2.0, beta: float = 4.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.beta  = beta

    def forward(self, pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
        # pred: [B, C, H_map, W_map]  — raw logits from the head
        # gt  : [B, C, H_map, W_map]  — Gaussian heatmap, values ∈ [0, 1]

        # Apply sigmoid and clamp to avoid log(0) instability
        p = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1.0 - 1e-4)
        # p: [B, C, H_map, W_map]

        pos_mask    = (gt == 1.0).float()               # [B, C, H_map, W_map]
        neg_mask    = 1.0 - pos_mask                    # [B, C, H_map, W_map]
        neg_weights = torch.pow(1.0 - gt, self.beta)   # [B, C, H_map, W_map]

        # Positive focal loss  — large at low-confidence peaks
        pos_loss = pos_mask * torch.pow(1.0 - p, self.alpha) * torch.log(p)
        # pos_loss: [B, C, H_map, W_map]

        # Negative focal loss — penalty-reduced near peaks
        neg_loss = neg_mask * neg_weights * torch.pow(p, self.alpha) * torch.log(1.0 - p)
        # neg_loss: [B, C, H_map, W_map]

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

    NMS via 2D MaxPool: a pixel survives iff it equals the local 3×3 maximum
    AND is above `threshold`.

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
        heatmap.unsqueeze(0),    # [1, 1, H_map, W_map]
        kernel_size=3,
        stride=1,
        padding=1,
    ).squeeze(0)                 # [1, H_map, W_map]

    # Keep local maxima above threshold
    keep = (heatmap == pooled) & (heatmap >= threshold)   # [1, H_map, W_map]

    scale_x = image_size / W_map
    scale_y = image_size / H_map

    indices = keep[0].nonzero(as_tuple=False)  # [N, 2] — (row, col)
    keypoints = [
        (float(col) * scale_x, float(row) * scale_y)
        for row, col in indices.tolist()
    ]
    return keypoints


def run_inference(
    model: nn.Module,
    image_path: str,
    cfg: Dict[str, Any],
    threshold: float = 0.3,
) -> List[Tuple[float, float]]:
    """
    End-to-end inference on a single image.
    Returns (x, y) keypoints in original image pixel coordinates.
    """
    device = torch.device(cfg["device"])
    model.eval()

    img_bgr = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    transform = A.Compose([
        A.Resize(height=cfg["image_size"], width=cfg["image_size"]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    img_t = transform(image=img_rgb)["image"].unsqueeze(0).to(device)
    # img_t: [1, 3, H, W]

    with torch.no_grad():
        raw_out  = model(img_t)                         # [1, 1, H_map, W_map]
        heatmap  = torch.sigmoid(raw_out).squeeze(0)   # [1, H_map, W_map]

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

    for step, (images, heatmaps) in enumerate(loader, start=1):
        images   = images.to(device, non_blocking=True)    # [B, 3, H, W]
        heatmaps = heatmaps.to(device, non_blocking=True)  # [B, 1, H_map, W_map]

        optimizer.zero_grad(set_to_none=True)
        pred = model(images)                               # [B, 1, H_map, W_map]
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
    for images, heatmaps in loader:
        images   = images.to(device, non_blocking=True)
        heatmaps = heatmaps.to(device, non_blocking=True)
        pred  = model(images)                              # [B, 1, H_map, W_map]
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
    patch_grid   = cfg["image_size"] // cfg["patch_size"]          # 16
    heatmap_size = patch_grid * (2 ** cfg["num_decoder_blocks"])   # 64

    print(
        f"Image {cfg['image_size']}px  |  "
        f"Patch grid {patch_grid}×{patch_grid}  |  "
        f"Heatmap {heatmap_size}×{heatmap_size}  |  "
        f"Train {n_train}  Val {n_val}"
    )

    # ── Datasets ──────────────────────────────────────────────────────────────
    train_ds = RoomCornerDataset(
        samples=train_samples,
        data_dir=cfg["data_dir"],
        image_size=cfg["image_size"],
        heatmap_size=heatmap_size,
        sigma=cfg["sigma"],
        augment=True,
    )
    val_ds = RoomCornerDataset(
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
    model = KeypointDINO(
        dino_channels=cfg["dino_channels"],
        neck_channels=cfg["neck_channels"],
        num_decoder_blocks=cfg["num_decoder_blocks"],
        num_classes=cfg["num_classes"],
        freeze_backbone=cfg["freeze_backbone"],
    ).to(device)

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(
        f"Trainable tensors: {len(trainable)}  |  "
        f"Parameters: {sum(p.numel() for p in trainable):,}"
    )

    # ── Optimizer (neck + decoder + head only) ────────────────────────────────
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