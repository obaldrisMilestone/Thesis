"""
Temporal depth aggregation visualization.

Produces three panel images for a chosen camera:
  depth_a_first_frame.png   — D_acc: first-frame disparity (pedestrian = bright blob)
  depth_b_last_frame.png    — D_new: last-frame disparity (pedestrian moved, wall revealed)
  depth_c_aggregated.png    — P_diff result: pedestrian erased, clean static depth

All images are saved to plots/ next to this script.
Colourmap: MAGMA (bright = high disparity = close; dark = far).
All three panels share the same global normalisation range for comparability.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import torch
from PIL import Image

from depth.depth_anything_simp import TemporalDepthSegmenter

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CAMERA_ID      = "Camera_31"   # large close-up pedestrian in frame 0, fully moved by end
BASE_DIR       = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VIDEO_PATH     = os.path.join(BASE_DIR, f"dataset/MTMC_Tracking_2025/val/Hospital_000/videos/{CAMERA_ID}.mp4")
OUTPUT_DIR     = os.path.dirname(os.path.abspath(__file__))   # plots/
DIFF_THRESHOLD = 0.15   # mirrors TemporalDepthSegmenter default
EMA_ALPHA      = 0.1    # mirrors TemporalDepthSegmenter default
# ---------------------------------------------------------------------------


def infer_disparity(segmenter: TemporalDepthSegmenter, bgr_frame: np.ndarray) -> np.ndarray:
    """Run DepthAnything on a single BGR frame; return raw disparity (float32 HxW)."""
    rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    inputs = segmenter.image_processor(images=pil, return_tensors="pt").to(segmenter.device)
    with torch.no_grad():
        outputs = segmenter.model(**inputs)
        pred = outputs.predicted_depth
    pred = torch.nn.functional.interpolate(
        pred.unsqueeze(1),
        size=pil.size[::-1],
        mode="bicubic",
        align_corners=False,
    ).squeeze()
    return pred.cpu().numpy().astype(np.float32)


def apply_pdiff(d_first: np.ndarray, d_last: np.ndarray,
                diff_threshold: float, alpha: float) -> np.ndarray:
    """
    Replicate the P_diff temporal aggregation from TemporalDepthSegmenter.process_video_depth.
    Returns the aggregated disparity map.
    """
    accumulated = d_first.copy()
    proportional_diff = (accumulated - d_last) / (accumulated + 1e-5)

    reveal_mask = proportional_diff > diff_threshold          # object moved away
    noise_mask  = np.abs(proportional_diff) <= diff_threshold # stable background
    # Condition C (spike: new object in front) — do nothing, keep D_first

    accumulated[reveal_mask] = d_last[reveal_mask]
    accumulated[noise_mask] = (
        (1.0 - alpha) * accumulated[noise_mask] + alpha * d_last[noise_mask]
    )
    return accumulated


def disparity_to_colormap(disp: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    """Normalise disparity to [0,255] with shared range and apply MAGMA colormap."""
    norm = np.clip((disp - vmin) / (vmax - vmin + 1e-8), 0.0, 1.0)
    gray = (norm * 255).astype(np.uint8)
    return cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)


def main():
    print("Loading DepthAnything model...")
    segmenter = TemporalDepthSegmenter()

    cap = cv2.VideoCapture(VIDEO_PATH)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Reading first frame from {VIDEO_PATH}")
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, frame_first = cap.read()
    assert ret, "Could not read first frame"

    print(f"Reading last frame (frame {total - 1})")
    cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
    ret, frame_last = cap.read()
    assert ret, "Could not read last frame"
    cap.release()

    print("Inferring disparity for first frame...")
    d_first = infer_disparity(segmenter, frame_first)

    print("Inferring disparity for last frame...")
    d_last = infer_disparity(segmenter, frame_last)

    print("Applying P_diff temporal aggregation...")
    d_agg = apply_pdiff(d_first, d_last, DIFF_THRESHOLD, EMA_ALPHA)

    # Shared normalisation range across all three maps for fair comparison
    vmin = min(d_first.min(), d_last.min(), d_agg.min())
    vmax = max(d_first.max(), d_last.max(), d_agg.max())
    print(f"Global disparity range: [{vmin:.3f}, {vmax:.3f}]")

    # --- Panel (a): first-frame disparity (D_acc) ---
    out_a = os.path.join(OUTPUT_DIR, "depth_a_first_frame.png")
    cv2.imwrite(out_a, disparity_to_colormap(d_first, vmin, vmax))
    print(f"Saved {out_a}")

    # --- Panel (b): last-frame disparity (D_new) ---
    out_b = os.path.join(OUTPUT_DIR, "depth_b_last_frame.png")
    cv2.imwrite(out_b, disparity_to_colormap(d_last, vmin, vmax))
    print(f"Saved {out_b}")

    # --- Panel (c): aggregated disparity after P_diff ---
    out_c = os.path.join(OUTPUT_DIR, "depth_c_aggregated.png")
    cv2.imwrite(out_c, disparity_to_colormap(d_agg, vmin, vmax))
    print(f"Saved {out_c}")

    print("Done.")


if __name__ == "__main__":
    main()
