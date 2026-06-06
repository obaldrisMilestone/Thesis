"""
KeypointDETR – DETR fine-tuned for keypoint detection on indoor scenes.

Replaces the bounding-box head with a 3-layer Keypoint MLP that predicts
(x, y, visibility) per keypoint slot.  Matching and losses are adapted
accordingly (no bbox / GIoU terms anywhere).

Script layout
─────────────
  1. KeypointDETR model definitions
  2. KeypointHungarianMatcher
  3. KeypointSetCriterion
  4. Mock indoor dataset (floor-wall intersection keypoints)
  5. main() training loop
"""

import math
import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader, Dataset

# ══════════════════════════════════════════════════════════════════════════════
# 1 ─ KeypointDETR Model Definitions
# ══════════════════════════════════════════════════════════════════════════════


class PositionalEncoding2D(nn.Module):
    """Fixed sinusoidal 2-D positional encoding.  No trainable parameters."""

    def __init__(self, d_model: int, temperature: float = 10_000.0):
        super().__init__()
        self.d_model     = d_model
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x : [B, d_model, H, W]
        B, _, H, W = x.shape
        device = x.device

        y_pos = torch.arange(H, device=device, dtype=torch.float32)  # [H]
        x_pos = torch.arange(W, device=device, dtype=torch.float32)  # [W]

        # Normalise to [0, 2π]
        y_pos = y_pos / H * 2.0 * math.pi
        x_pos = x_pos / W * 2.0 * math.pi

        # Each axis gets d_model//2 channels encoded as sin/cos pairs
        half = self.d_model // 4   # y contributes half*2 dims, x another half*2
        dim_t = self.temperature ** (
            2.0 * torch.arange(half, device=device, dtype=torch.float32) / half
        )                                                          # [half]

        y_enc = y_pos.unsqueeze(1) / dim_t                        # [H, half]
        x_enc = x_pos.unsqueeze(1) / dim_t                        # [W, half]

        # Interleave sin/cos → [H, d_model//2] and [W, d_model//2]
        y_enc = torch.stack([y_enc.sin(), y_enc.cos()], dim=-1).flatten(-2)   # [H, 2*half]
        x_enc = torch.stack([x_enc.sin(), x_enc.cos()], dim=-1).flatten(-2)   # [W, 2*half]

        # Broadcast over both spatial axes
        y_enc = y_enc.unsqueeze(1).expand(H, W, -1)   # [H, W, d_model//2]
        x_enc = x_enc.unsqueeze(0).expand(H, W, -1)   # [H, W, d_model//2]

        pos = torch.cat([y_enc, x_enc], dim=-1)        # [H, W, d_model]
        pos = pos.permute(2, 0, 1).unsqueeze(0)        # [1, d_model, H, W]
        return pos.expand(B, -1, -1, -1)               # [B, d_model, H, W]


class Backbone(nn.Module):
    """
    ResNet-50 feature extractor with 1×1 channel projection.

    Output spatial stride = 32, so a 224×224 input yields a 7×7 feature map.
    """

    def __init__(self, d_model: int = 256, pretrained: bool = True):
        super().__init__()
        weights   = tvm.ResNet50_Weights.DEFAULT if pretrained else None
        resnet    = tvm.resnet50(weights=weights)
        # Drop avgpool and fc; keep everything up to layer4
        self.body = nn.Sequential(*list(resnet.children())[:-2])  # → [B, 2048, H/32, W/32]
        self.proj = nn.Conv2d(2048, d_model, kernel_size=1)        # → [B, d_model, H/32, W/32]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x    : [B, 3, H, W]
        feat = self.body(x)    # [B, 2048, H/32, W/32]
        feat = self.proj(feat) # [B, d_model, H/32, W/32]
        return feat


class DETRTransformer(nn.Module):
    """Standard Transformer Encoder-Decoder as used in DETR."""

    def __init__(
        self,
        d_model:            int   = 256,
        nhead:              int   = 8,
        num_encoder_layers: int   = 6,
        num_decoder_layers: int   = 6,
        dim_feedforward:    int   = 2048,
        dropout:            float = 0.1,
    ):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_encoder_layers)

        dec_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_decoder_layers)

    def forward(
        self,
        src:         torch.Tensor,  # [B, d_model, H, W]  backbone features
        pos:         torch.Tensor,  # [B, d_model, H, W]  spatial positional encoding
        query_embed: torch.Tensor,  # [Q, d_model]         learned object queries
    ) -> torch.Tensor:
        B, C, H, W = src.shape

        src_seq  = src.flatten(2).permute(0, 2, 1)   # [B, H*W, d_model]
        pos_seq  = pos.flatten(2).permute(0, 2, 1)   # [B, H*W, d_model]
        memory   = self.encoder(src_seq + pos_seq)   # [B, H*W, d_model]

        # Object queries serve as both target content and positional signals
        tgt = query_embed.unsqueeze(0).expand(B, -1, -1)  # [B, Q, d_model]
        hs  = self.decoder(tgt, memory)                    # [B, Q, d_model]
        return hs


class MLP(nn.Module):
    """3-layer MLP: Linear → ReLU → Linear → ReLU → Linear."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class KeypointDETR(nn.Module):
    """
    DETR adapted for keypoint detection.

    Heads
    ─────
    class_predictor   : Linear(d_model, num_classes + 1)
    keypoint_predictor: 3-layer MLP(d_model, d_model, num_keypoints * 3)
                        outputs (x, y) as sigmoid-normalised coords + raw visibility logit
    """

    def __init__(
        self,
        num_classes:        int,
        num_queries:        int,
        num_keypoints:      int,
        d_model:            int   = 256,
        nhead:              int   = 8,
        num_encoder_layers: int   = 6,
        num_decoder_layers: int   = 6,
        dim_feedforward:    int   = 2048,
        dropout:            float = 0.1,
        pretrained_backbone: bool = True,
    ):
        super().__init__()
        self.num_keypoints = num_keypoints

        self.backbone     = Backbone(d_model, pretrained=pretrained_backbone)
        self.pos_encoding = PositionalEncoding2D(d_model)
        self.transformer  = DETRTransformer(
            d_model, nhead,
            num_encoder_layers, num_decoder_layers,
            dim_feedforward, dropout,
        )
        self.query_embed = nn.Embedding(num_queries, d_model)

        # Classification head: retains background slot at index num_classes
        self.class_predictor    = nn.Linear(d_model, num_classes + 1)

        # Keypoint head: replaces bbox head; outputs K * 3 values per query
        self.keypoint_predictor = MLP(d_model, d_model, num_keypoints * 3)

    def forward(self, images: torch.Tensor) -> dict:
        # images : [B, 3, H, W]
        B = images.shape[0]

        features = self.backbone(images)           # [B, d_model, H/32, W/32]
        pos      = self.pos_encoding(features)     # [B, d_model, H/32, W/32]
        hs       = self.transformer(features, pos, self.query_embed.weight)
        #                                          # [B, Q, d_model]

        pred_logits = self.class_predictor(hs)     # [B, Q, num_classes + 1]

        kp_raw = self.keypoint_predictor(hs)       # [B, Q, K*3]
        kp_raw = kp_raw.view(B, -1, self.num_keypoints, 3)  # [B, Q, K, 3]

        pred_xy  = torch.sigmoid(kp_raw[..., :2])  # [B, Q, K, 2]  – normalised [0, 1]
        pred_vis = kp_raw[..., 2:3]                 # [B, Q, K, 1]  – raw visibility logit

        pred_keypoints = torch.cat([pred_xy, pred_vis], dim=-1)  # [B, Q, K, 3]

        return {
            "pred_logits":    pred_logits,    # [B, Q, num_classes + 1]
            "pred_keypoints": pred_keypoints, # [B, Q, K, 3]
        }


# ══════════════════════════════════════════════════════════════════════════════
# 2 ─ KeypointHungarianMatcher
# ══════════════════════════════════════════════════════════════════════════════


class KeypointHungarianMatcher(nn.Module):
    """
    Bipartite matching between predicted queries and ground-truth instances.

    Cost matrix components
    ──────────────────────
    - Classification  : negative softmax probability for the GT class
    - Keypoint L1     : mean L1 on (x, y), masked to GT-visible keypoints only
    No bounding-box or GIoU terms are used.
    """

    def __init__(self, cost_class: float = 1.0, cost_keypoint: float = 5.0):
        super().__init__()
        self.cost_class    = cost_class
        self.cost_keypoint = cost_keypoint

    @torch.no_grad()
    def forward(self, outputs: dict, targets: list) -> list:
        """
        Returns
        ───────
        List of (src_idx, tgt_idx) pairs – one per image in the batch.
        Both tensors are 1-D LongTensors of the same length M (matched pairs).
        """
        B = outputs["pred_logits"].shape[0]
        indices = []

        for b in range(B):
            tgt_labels    = targets[b]["labels"]     # [N]
            tgt_keypoints = targets[b]["keypoints"]  # [N, K, 3]
            N = len(tgt_labels)

            if N == 0:
                indices.append((
                    torch.tensor([], dtype=torch.long),
                    torch.tensor([], dtype=torch.long),
                ))
                continue

            # ── Classification cost ──────────────────────────────────────────
            out_prob  = F.softmax(outputs["pred_logits"][b], dim=-1)  # [Q, C+1]
            cost_cls  = -out_prob[:, tgt_labels]                       # [Q, N]

            # ── Keypoint L1 cost (masked to labeled GT keypoints) ────────────
            pred_xy = outputs["pred_keypoints"][b, :, :, :2]  # [Q, K, 2]
            gt_xy   = tgt_keypoints[:, :, :2]                  # [N, K, 2]
            gt_vis  = tgt_keypoints[:, :, 2]                   # [N, K]

            # vis_mask[n, k] == True  ↔  GT keypoint k of instance n is labeled
            vis_mask = (gt_vis > 0).float()  # [N, K]

            # Pairwise L1: [Q, 1, K, 2] vs [1, N, K, 2]  →  [Q, N, K]
            l1 = (pred_xy.unsqueeze(1) - gt_xy.unsqueeze(0)).abs().sum(-1)  # [Q, N, K]

            # Zero-out unlabeled keypoints; normalise by visible count per GT
            num_vis   = vis_mask.sum(-1).clamp(min=1.0)              # [N]
            cost_kp   = (l1 * vis_mask.unsqueeze(0)).sum(-1)         # [Q, N]
            cost_kp   = cost_kp / num_vis.unsqueeze(0)               # [Q, N]

            # ── Total cost ───────────────────────────────────────────────────
            C = self.cost_class * cost_cls + self.cost_keypoint * cost_kp  # [Q, N]

            row_ind, col_ind = linear_sum_assignment(C.cpu().numpy())
            indices.append((
                torch.as_tensor(row_ind, dtype=torch.long),
                torch.as_tensor(col_ind, dtype=torch.long),
            ))

        return indices


# ══════════════════════════════════════════════════════════════════════════════
# 3 ─ KeypointSetCriterion
# ══════════════════════════════════════════════════════════════════════════════


class KeypointSetCriterion(nn.Module):
    """
    Three-term loss applied after Hungarian matching.

    1. Classification  – cross-entropy over all Q queries (unmatched → background)
    2. Coordinate      – Smooth-L1 on (x, y) for matched queries & visible keypoints
    3. Visibility      – BCE on the visibility logit for matched queries
    """

    def __init__(
        self,
        num_classes:     int,
        matcher:         KeypointHungarianMatcher,
        weight_class:    float = 1.0,
        weight_kp_coord: float = 5.0,
        weight_kp_vis:   float = 1.0,
    ):
        super().__init__()
        self.num_classes     = num_classes
        self.matcher         = matcher
        self.weight_class    = weight_class
        self.weight_kp_coord = weight_kp_coord
        self.weight_kp_vis   = weight_kp_vis

    # ── Individual losses ────────────────────────────────────────────────────

    def _loss_labels(
        self, outputs: dict, targets: list, indices: list
    ) -> torch.Tensor:
        pred_logits = outputs["pred_logits"]   # [B, Q, C+1]
        B, Q, _     = pred_logits.shape
        device      = pred_logits.device

        # Default every query slot to the background class index
        tgt_classes = torch.full(
            (B, Q), self.num_classes, dtype=torch.long, device=device
        )                                      # [B, Q]

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                tgt_classes[b, src_idx] = targets[b]["labels"][tgt_idx]  # [M]

        # Reshape for F.cross_entropy
        loss = F.cross_entropy(
            pred_logits.reshape(B * Q, -1),    # [B*Q, C+1]
            tgt_classes.reshape(B * Q),        # [B*Q]
        )
        return loss

    def _loss_kp_coord(
        self, outputs: dict, targets: list, indices: list
    ) -> torch.Tensor:
        pred_kp     = outputs["pred_keypoints"]  # [B, Q, K, 3]
        device      = pred_kp.device
        total_loss  = torch.tensor(0.0, device=device)
        total_valid = 0

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue

            pred_xy = pred_kp[b, src_idx, :, :2]           # [M, K, 2]
            gt_kps  = targets[b]["keypoints"][tgt_idx]      # [M, K, 3]
            gt_xy   = gt_kps[:, :, :2]                      # [M, K, 2]
            gt_vis  = gt_kps[:, :, 2]                       # [M, K]

            vis_mask  = (gt_vis > 0).float()                 # [M, K]
            n_vis     = int(vis_mask.sum().item())
            if n_vis == 0:
                continue

            smooth_l1 = F.smooth_l1_loss(pred_xy, gt_xy, reduction="none")  # [M, K, 2]
            smooth_l1 = smooth_l1.sum(-1)                                    # [M, K]
            smooth_l1 = smooth_l1 * vis_mask                                 # [M, K]  masked

            total_loss  = total_loss + smooth_l1.sum()
            total_valid += n_vis

        return total_loss / max(total_valid, 1)

    def _loss_kp_vis(
        self, outputs: dict, targets: list, indices: list
    ) -> torch.Tensor:
        pred_kp    = outputs["pred_keypoints"]   # [B, Q, K, 3]
        device     = pred_kp.device
        total_loss = torch.tensor(0.0, device=device)
        total_kps  = 0

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue

            pred_vis_logit = pred_kp[b, src_idx, :, 2]               # [M, K]
            gt_vis_raw     = targets[b]["keypoints"][tgt_idx, :, 2]   # [M, K]

            # Fully visible (2) → 1.0 ; occluded (1) or unlabeled (0) → 0.0
            gt_vis_binary = (gt_vis_raw == 2).float()                 # [M, K]

            total_loss = total_loss + F.binary_cross_entropy_with_logits(
                pred_vis_logit, gt_vis_binary, reduction="sum"
            )
            total_kps += pred_vis_logit.numel()

        return total_loss / max(total_kps, 1)

    # ── Combined forward ──────────────────────────────────────────────────────

    def forward(self, outputs: dict, targets: list) -> dict:
        indices = self.matcher(outputs, targets)

        l_cls   = self._loss_labels(outputs, targets, indices)
        l_coord = self._loss_kp_coord(outputs, targets, indices)
        l_vis   = self._loss_kp_vis(outputs, targets, indices)

        total = (
            self.weight_class    * l_cls   +
            self.weight_kp_coord * l_coord +
            self.weight_kp_vis   * l_vis
        )

        return {
            "loss":       total,
            "loss_class": l_cls.detach(),
            "loss_coord": l_coord.detach(),
            "loss_vis":   l_vis.detach(),
        }


# ══════════════════════════════════════════════════════════════════════════════
# 4 ─ Mock Dataset – Indoor Scenes with Geometric Keypoints
# ══════════════════════════════════════════════════════════════════════════════


class IndoorKeypointDataset(Dataset):
    """
    Procedurally generated indoor-scene dataset.

    Each image simulates a perspective room view: a warm floor region below a
    horizon line and cooler walls above it.  Keypoints are placed along the
    horizon, approximating floor-wall intersection corners visible from the
    camera.

    Returns
    ───────
    image  : FloatTensor [3, H, W]
    target : dict
        labels    : LongTensor  [N]        – class ID per instance
        keypoints : FloatTensor [N, K, 3]  – (x_norm, y_norm, visibility∈{0,1,2})
    """

    def __init__(
        self,
        num_samples:   int = 200,
        img_size:      int = 224,
        num_keypoints: int = 4,
        num_classes:   int = 1,
        max_instances: int = 3,
        seed:          int = 0,
    ):
        super().__init__()
        self.num_samples   = num_samples
        self.img_size      = img_size
        self.num_keypoints = num_keypoints
        self.num_classes   = num_classes
        self.max_instances = max_instances

        rng = random.Random(seed)
        np.random.seed(seed)
        self.samples = [self._generate(rng) for _ in range(num_samples)]

    def _generate(self, rng: random.Random) -> tuple:
        H = W = self.img_size

        # ── Synthetic image ─────────────────────────────────────────────────
        horizon_px  = int(H * rng.uniform(0.38, 0.56))
        floor_h     = H - horizon_px

        img = np.zeros((3, H, W), dtype=np.float32)

        # Floor: warm gradient (R heavy), slightly noisy
        for c, base, end in [(0, 0.48, 0.30), (1, 0.36, 0.22), (2, 0.20, 0.12)]:
            grad = np.linspace(base, end, floor_h).reshape(-1, 1)
            img[c, horizon_px:, :] = grad + np.random.randn(floor_h, W) * 0.015

        # Walls: cool gradient (B/G heavy), slightly noisy
        for c, base, end in [(0, 0.68, 0.58), (1, 0.68, 0.58), (2, 0.78, 0.65)]:
            grad = np.linspace(base, end, horizon_px).reshape(-1, 1)
            img[c, :horizon_px, :] = grad + np.random.randn(horizon_px, W) * 0.025

        img = np.clip(img, 0.0, 1.0)
        image_t = torch.from_numpy(img)  # [3, H, W]

        # ── Instances (floor-wall intersection regions) ──────────────────────
        n              = rng.randint(1, self.max_instances)
        horizon_y_norm = horizon_px / H

        labels_list    = []
        keypoints_list = []

        for _ in range(n):
            kps = []
            for _ in range(self.num_keypoints):
                kp_x = rng.uniform(0.05, 0.95)
                # Keypoints cluster ±6 % around the floor-wall horizon
                kp_y = horizon_y_norm + rng.uniform(-0.06, 0.06)
                kp_y = max(0.02, min(0.98, kp_y))
                # Visibility: 2=visible (80 %), 1=occluded (15 %), 0=unlabeled (5 %)
                vis  = rng.choices([2, 1, 0], weights=[0.80, 0.15, 0.05])[0]
                kps.append([kp_x, kp_y, float(vis)])

            labels_list.append(rng.randint(0, self.num_classes - 1))
            keypoints_list.append(kps)

        labels_t    = torch.tensor(labels_list,    dtype=torch.long)
        keypoints_t = torch.tensor(keypoints_list, dtype=torch.float32)  # [N, K, 3]

        return image_t, {"labels": labels_t, "keypoints": keypoints_t}

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple:
        return self.samples[idx]


def collate_fn(batch: list) -> tuple:
    """Stack images into a batch tensor; keep targets as a list of dicts."""
    images  = torch.stack([item[0] for item in batch])  # [B, 3, H, W]
    targets = [item[1] for item in batch]
    return images, targets


# ══════════════════════════════════════════════════════════════════════════════
# 5 ─ main() – Training Loop
# ══════════════════════════════════════════════════════════════════════════════


def main() -> None:
    # ── Hyperparameters ──────────────────────────────────────────────────────
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model
    NUM_CLASSES        = 1     # single semantic class: "interior region"
    NUM_QUERIES        = 20    # object query slots
    NUM_KEYPOINTS      = 4     # floor-wall corners per instance
    D_MODEL            = 256
    NHEAD              = 8
    NUM_ENC_LAYERS     = 3     # kept small for tractable mock training
    NUM_DEC_LAYERS     = 3
    DIM_FEEDFORWARD    = 1024
    DROPOUT            = 0.1
    PRETRAINED_BACKBONE = True

    # Learning rates
    FREEZE_BACKBONE    = False  # set True to lock backbone.body weights
    LR_BACKBONE        = 1e-5  # very low LR for CNN backbone
    LR_BASE            = 1e-4  # base LR for transformer + heads
    WEIGHT_DECAY       = 1e-4

    # Training loop
    BATCH_SIZE         = 4
    NUM_EPOCHS         = 30
    GRAD_CLIP_NORM     = 0.1
    LR_STEP_SIZE       = 10    # decay LR every N epochs
    LR_GAMMA           = 0.5

    # Dataset
    IMG_SIZE           = 224
    NUM_TRAIN          = 200
    NUM_VAL            = 40

    print(f"Device        : {DEVICE}")
    print(f"Queries/Keypts: {NUM_QUERIES} / {NUM_KEYPOINTS}")
    print(f"d_model       : {D_MODEL}  |  enc={NUM_ENC_LAYERS}  dec={NUM_DEC_LAYERS}")
    print()

    # ── Datasets & loaders ───────────────────────────────────────────────────
    train_ds = IndoorKeypointDataset(
        NUM_TRAIN, IMG_SIZE, NUM_KEYPOINTS, NUM_CLASSES, seed=0
    )
    val_ds = IndoorKeypointDataset(
        NUM_VAL, IMG_SIZE, NUM_KEYPOINTS, NUM_CLASSES, seed=1
    )

    train_loader = DataLoader(
        train_ds, BATCH_SIZE, shuffle=True,  collate_fn=collate_fn, num_workers=0
    )
    val_loader = DataLoader(
        val_ds,   BATCH_SIZE, shuffle=False, collate_fn=collate_fn, num_workers=0
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = KeypointDETR(
        num_classes         = NUM_CLASSES,
        num_queries         = NUM_QUERIES,
        num_keypoints       = NUM_KEYPOINTS,
        d_model             = D_MODEL,
        nhead               = NHEAD,
        num_encoder_layers  = NUM_ENC_LAYERS,
        num_decoder_layers  = NUM_DEC_LAYERS,
        dim_feedforward     = DIM_FEEDFORWARD,
        dropout             = DROPOUT,
        pretrained_backbone = PRETRAINED_BACKBONE,
    ).to(DEVICE)

    # ── Optionally freeze backbone body ──────────────────────────────────────
    if FREEZE_BACKBONE:
        for p in model.backbone.body.parameters():
            p.requires_grad = False
        print("Backbone body frozen (backbone.proj and all other params trainable).")

    # ── Differential parameter groups ────────────────────────────────────────
    # backbone (body + proj) → low LR; transformer + heads → base LR
    backbone_params = [p for p in model.backbone.parameters() if p.requires_grad]

    other_params = (
        list(model.transformer.parameters())       +  # encoder + decoder
        list(model.class_predictor.parameters())   +  # classification head
        list(model.keypoint_predictor.parameters()) +  # keypoint MLP head
        [model.query_embed.weight]                    # learned object queries
        # pos_encoding has no trainable parameters (fixed sinusoidal)
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_params, "lr": LR_BACKBONE},
            {"params": other_params,    "lr": LR_BASE},
        ],
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA
    )

    # ── Loss ─────────────────────────────────────────────────────────────────
    matcher   = KeypointHungarianMatcher(cost_class=1.0, cost_keypoint=5.0)
    criterion = KeypointSetCriterion(
        num_classes     = NUM_CLASSES,
        matcher         = matcher,
        weight_class    = 1.0,
        weight_kp_coord = 5.0,
        weight_kp_vis   = 1.0,
    ).to(DEVICE)

    # ── Epoch loop ───────────────────────────────────────────────────────────
    for epoch in range(1, NUM_EPOCHS + 1):

        # ── Train ────────────────────────────────────────────────────────────
        model.train()
        tr = {"loss": 0.0, "loss_class": 0.0, "loss_coord": 0.0, "loss_vis": 0.0}

        for images, targets in train_loader:
            images  = images.to(DEVICE)                                           # [B, 3, H, W]
            targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

            outputs   = model(images)
            loss_dict = criterion(outputs, targets)

            optimizer.zero_grad(set_to_none=True)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()

            for k in tr:
                tr[k] += loss_dict[k].item()

        scheduler.step()
        n_tr = len(train_loader)

        # ── Evaluate ─────────────────────────────────────────────────────────
        model.eval()
        vl = {"loss": 0.0, "loss_class": 0.0, "loss_coord": 0.0, "loss_vis": 0.0}

        with torch.no_grad():
            for images, targets in val_loader:
                images  = images.to(DEVICE)
                targets = [{k: v.to(DEVICE) for k, v in t.items()} for t in targets]

                outputs   = model(images)
                loss_dict = criterion(outputs, targets)

                for k in vl:
                    vl[k] += loss_dict[k].item()

        n_vl = len(val_loader)

        print(
            f"Epoch [{epoch:03d}/{NUM_EPOCHS}]  "
            f"Train loss={tr['loss']/n_tr:.4f} "
            f"(cls={tr['loss_class']/n_tr:.3f} "
            f"coord={tr['loss_coord']/n_tr:.3f} "
            f"vis={tr['loss_vis']/n_tr:.3f})  |  "
            f"Val loss={vl['loss']/n_vl:.4f} "
            f"(cls={vl['loss_class']/n_vl:.3f} "
            f"coord={vl['loss_coord']/n_vl:.3f} "
            f"vis={vl['loss_vis']/n_vl:.3f})"
        )


if __name__ == "__main__":
    main()
