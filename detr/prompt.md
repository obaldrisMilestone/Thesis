<role>
You are an Expert Computer Vision and PyTorch Deep Learning Engineer. 
</role>

<task>
Write a complete, clean, and modular PyTorch training script to fine-tune a DEtection TRansformer (DETR) model modified specifically for Keypoint Detection instead of Object Detection (bounding boxes). 
</task>

<architecture_modifications>
1. Backbone & Transformer: Use a standard DETR architecture (e.g., using `torchvision.models.detection.detr` or `timm`). The CNN Backbone and Transformer Encoder/Decoder configurations must remain standard.
2. Prediction Heads: 
   - Replace the original bounding box MLP head (`bbox_predictor`) with a new Keypoint MLP head.
   - The Keypoint head MUST be a 3-layer MLP with ReLU activations outputting a tensor of shape `[batch_size, num_queries, num_keypoints * 3]`. 
   - For each keypoint, predict (x, y, visibility). `x` and `y` are normalized coordinates [0, 1] relative to image size. `visibility` is a logit indicating keypoint presence.
   - Retain the classification head (`class_predictor`) outputting `[batch_size, num_queries, num_classes + 1]` for object/background classification.
</architecture_modifications>

<data_format_specs>
The custom Dataset must return:
- `images`: Tensor of shape `[3, H, W]`
- `targets`: A dictionary containing:
  - `labels`: Tensor of shape `[num_target_boxes]` (Class IDs)
  - `keypoints`: Tensor of shape `[num_target_boxes, num_keypoints, 3]`. Format is `[x, y, visibility_flag]`. 
  - Visibility mapping: 0=unlabeled, 1=labeled but occluded, 2=labeled and visible.
</data_format_specs>

<hungarian_matcher_modifications>
Update the `HungarianMatcher` to compute the bipartite assignment cost matrix using ONLY:
1. Classification cost: Negative log-probabilities or cross-entropy cost.
2. Keypoint L1 cost: L1 distance between predicted (x, y) and ground-truth (x, y), strictly MASKED to penalize only keypoints where ground-truth visibility > 0.
CRITICAL: Do NOT use bounding box L1 loss or GIoU loss in the cost matrix.
</hungarian_matcher_modifications>

<loss_function_modifications>
Implement a custom `SetCriterion` calculating:
1. Classification Loss: Cross-entropy across all queries vs. matched targets (including background).
2. Keypoint Coordinate Loss: L1 or Smooth L1 loss between predicted and ground-truth (x, y). Calculate ONLY for matched queries and ONLY for keypoints where ground-truth visibility > 0.
3. Keypoint Visibility Loss: BCE or Cross-Entropy loss for the visibility state of the keypoints.
</loss_function_modifications>

<training_specifications>
- Parameter Groups & Freezing: 
  - CNN backbone parameters MUST use a very low learning rate (1e-5) or be frozen via a toggle config.
  - Transformer Encoder/Decoder MUST use the base learning rate (1e-4).
  - New Keypoint MLP and Class MLP heads MUST use the base learning rate (1e-4).
- Optimizer: AdamW with weight decay (1e-4).
- Loop Setup: Include gradient clipping (max norm = 0.1), learning rate scheduling (StepLR), an evaluation step, and console metric logging (Loss/Epoch).
</training_specifications>

<code_requirements>
- NO PLACEHOLDERS. Do not use "omitted for brevity" comments. Write out full classes, methods, and functions.
- Include explicit tensor shape comments (e.g., `# [B, Q, K*3]`) at critical transformations inside forward passes and loss functions.
- Structure the script sequentially into these 5 components:
  1. `KeypointDETR` model definitions.
  2. `KeypointHungarianMatcher`.
  3. `KeypointSetCriterion`.
  4. Mock Dataset Generation (simulate indoor scenes with geometric keypoints like floor-wall intersections so the script runs out-of-the-box).
  5. The primary `main()` training loop block.
</code_requirements>