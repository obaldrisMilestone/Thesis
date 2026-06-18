"""
Normalize depth maps to [0, 1] float32.

Reads all .npy depth maps from INPUT_DIR, normalizes each independently
using its own min/max range, and writes them to OUTPUT_DIR at the same
directory level as INPUT_DIR.

Each file is normalized as:
    depth_norm = depth / depth.max()

Scale-only (no shift) normalization is used intentionally: subtracting the
minimum before dividing adds a per-pixel offset to every unprojected 3D point,
which warps flat surfaces into curved ones and breaks RANSAC plane fitting.
"""

import os
import numpy as np

# ── configuration ──────────────────────────────────────────────────────────────
INPUT_DIR = '/home/user/thesis/code/dataset/MTMC_Tracking_2025/val/Hospital_000/depth_maps'
# OUTPUT_DIR is derived automatically: same parent, folder name = INPUT_DIR name + '_normalized'
# Override by setting OUTPUT_DIR to an explicit path.
OUTPUT_DIR = None
# ──────────────────────────────────────────────────────────────────────────────


def normalize_depth_maps(input_dir, output_dir=None):
    input_dir = os.path.abspath(input_dir)

    if output_dir is None:
        parent = os.path.dirname(input_dir)
        folder_name = os.path.basename(input_dir) + '_normalized'
        output_dir = os.path.join(parent, folder_name)

    os.makedirs(output_dir, exist_ok=True)

    npy_files = sorted(f for f in os.listdir(input_dir) if f.endswith('.npy'))
    if not npy_files:
        print(f"No .npy files found in {input_dir}")
        return

    print(f"Input  : {input_dir}")
    print(f"Output : {output_dir}")
    print(f"Files  : {len(npy_files)}\n")

    for fname in npy_files:
        src = os.path.join(input_dir, fname)
        dst = os.path.join(output_dir, fname)

        depth = np.load(src).astype(np.float32)

        d_max = float(depth.max())

        if d_max == 0:
            print(f"  {fname}  — all zeros, skipped")
            continue

        normalized = depth / d_max

        np.save(dst, normalized)
        print(f"  {fname}  dtype={depth.dtype}  max={d_max:.1f}  → [0, 1] float32")

    print(f"\nDone. {len(npy_files)} file(s) written to {output_dir}")


if __name__ == "__main__":
    normalize_depth_maps(INPUT_DIR, OUTPUT_DIR)
