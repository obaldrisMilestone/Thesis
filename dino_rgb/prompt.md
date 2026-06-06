# Coding Agent Specification: RGB-Only Room Corner Detection Architecture

**Context for the Coding Agent:**
You are an expert PyTorch computer vision engineer. Your task is to write a complete, modular, and well-documented PyTorch training pipeline for a dense prediction task (Keypoint Detection via Heatmaps). 

The goal of this model is to extract room corners (x, y coordinates) from indoor scenes using a single-stream architecture utilizing a frozen foundation model (DINOv2) to extract high-fidelity semantic and geometric features directly from RGB images. This script will serve as the baseline for an ablation study.

Please write the complete code implementing the specifications detailed below. Organize the code into logical Python modules (e.g., `dataset.py`, `model.py`, `train.py`, `inference.py`) or one comprehensive, highly-structured script.

---

## 1. Model Architecture Specification

The model takes a single input modality:
1. `rgb`: Tensor of shape `[B, 3, H, W]`

### A. The RGB Stream (Frozen Encoder)
* **Backbone:** `dinov2_vits14` (load via `torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')`).
* **Constraint:** This entire sub-network MUST be strictly frozen (`requires_grad = False`).
* **Forward Pass:** * Pass the RGB image through the backbone.
    * Extract the patch tokens (ignore the `[CLS]` token).
    * Reshape the 1D sequence of tokens into a 2D spatial grid. 
    * *Math Note:* For a $224 \times 224$ image with a patch size of $14$, the grid shape is `[B, 384, 16, 16]`. Dynamically calculate the grid size based on input dimensions `H / 14` and `W / 14`.

### B. The Neck (Bottleneck Layer)
* Pass the 2D feature grid from the RGB stream into a dimension-reduction layer.
* Apply a $1 \times 1$ Convolution + BatchNorm + ReLU to reduce the channel depth (e.g., from `384` for DINOv2-S) to a uniform decoder size (e.g., `256` channels) to manage computational footprint.

### C. The Decoder (Upsampling)
* Implement a series of upsampling blocks to rebuild spatial resolution. 
* **Block architecture:** `Bilinear Interpolation (scale_factor=2)` -> `3x3 Conv2d (padding=1)` -> `BatchNorm2d` -> `ReLU`.
* Repeat these blocks until reaching the target heatmap resolution (e.g., `H/2, W/2` or `H/4, W/4` of the original image size).

### D. The Head
* A final $1 \times 1$ `Conv2d` layer that projects the features to `num_classes` channels (default `num_classes=1` for "any corner").
* *Note:* No Sigmoid activation is needed here if using raw logits directly in the focal loss computation, but ensure you apply it dynamically during loss/inference as detailed below.

---

## 2. Dataset & Dataloader Specification

Create a custom `torch.utils.data.Dataset`.

### A. Data Inputs
* Read RGB images (standardized to ImageNet mean/std).
* Read Keypoint annotations (a list of [x, y] coordinates for each image).

### B. Ground Truth Heatmap Generation (CRITICAL)
* Do not output raw coordinates. Write a function `generate_gaussian_heatmap(image_size, keypoints, sigma)` that generates the target tensor for training.
* For each [x, y] keypoint, draw an unnormalized 2D Gaussian blob on a blank single-channel image. The peak of the Gaussian (value 1.0) must be exactly at the (x, y) coordinate.
* The heatmap must match the exact spatial resolution of the Decoder's output (e.g., `[B, 1, H/4, W/4]`). Remember to scale down the GT coordinates by the same factor before drawing the Gaussians!

### C. Augmentations
* Use `albumentations` library to ensure coordinates are transformed perfectly alongside images.
* **Include:** `HorizontalFlip`, `RandomResizedCrop` (ensure aspect ratio doesn't stretch wildly), `ColorJitter`.
* **STRICTLY FORBIDDEN:** Rotations.

---

## 3. Training Loop Specification

* **Loss Function:** **Penalty-Reduced Focal Loss (CenterNet style)**. DO NOT use `nn.MSELoss`.
  * **Implementation requirement:** Write a custom `HeatmapFocalLoss(nn.Module)` with `alpha=2` and `beta=4`.
  * Apply `torch.clamp(torch.sigmoid(pred), min=1e-4, max=1-1e-4)` to the raw logits to bound them.
  * Use the ground truth Gaussian values as a penalty reduction weight for negative samples: `neg_weights = torch.pow(1 - gt, beta)`.
  * Compute standard focal loss for positive matches (where `gt == 1`) and penalty-reduced loss for background/nearby pixels.
* **Optimizer:** `torch.optim.AdamW`. **Ensure only the parameters of the Neck, Decoder, and Head are passed to the optimizer.** (Verify `len(list(filter(lambda p: p.requires_grad, model.parameters())))`).
* **LR Scheduler:** `CosineAnnealingLR` or `ReduceLROnPlateau`.
* Include standard training logging (Train Loss, Validation Loss).

---

## 4. Inference & Keypoint Extraction

Write an inference function that translates the model's heatmap back into raw $(x, y)$ coordinates.
* **Post-processing:** Pass the heatmap through Non-Maximum Suppression (NMS) using a 2D MaxPool operation (`kernel_size=3, stride=1, padding=1`). A pixel is a local maximum if `heatmap == maxpool(heatmap)`.
* Filter out peaks that fall below a confidence `threshold` (e.g., `0.3`).
* Return the scaled-up $(x, y)$ coordinates to match the original input image resolution.