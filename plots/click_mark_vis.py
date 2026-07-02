"""
Interactive point-marking tool.

Loads a camera image with its ground-truth annotated corners overlaid, then
lets the user click to simulate detected corner candidates.

Left-click  : add a detected marker
Right-click : remove the nearest detected marker
Press 'c'   : clear all detected markers
Press 's' or close the window : save and exit

GT annotations are shown as green diamonds (static, not interactive).
Detected (clicked) points are shown as yellow circles.

Output: plots/click_mark_out.png  (image + GT annotations + detected markers)
        prints (x, y) coordinates of each detected point to stdout
"""

import os
import json
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
CAMERA_ID    = "Camera_25"
BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMAGE_PATH   = os.path.join(BASE_DIR, "dataset/MTMC_Tracking_2025/val/Hospital_000/images", f"{CAMERA_ID}.jpg")
CAM_ANNS     = os.path.join(BASE_DIR, "dataset/MTMC_Tracking_2025/annotations/val/Hospital_000/annotated_cameras.json")
OUTPUT_PATH  = os.path.join(BASE_DIR, "plots/click_mark_out.png")

MARKER_RADIUS = 12      # radius for detected markers in saved image
GT_RADIUS     = 12      # radius for GT markers in saved image
# ---------------------------------------------------------------------------

# BGR colours
_GT_COLOR      = (0, 200, 0)     # green  — ground truth
_DET_COLOR     = (0, 200, 255)   # yellow — detected (clicked)
_LABEL_WHITE   = (255, 255, 255)


def _nearest_idx(points, x, y):
    if not points:
        return -1
    dists = [(px - x) ** 2 + (py - y) ** 2 for px, py in points]
    return int(np.argmin(dists))


def _load_gt_annotations(cam_anns_path, camera_id):
    """Return list of {point_id, x, y} for this camera, or []."""
    if not os.path.exists(cam_anns_path):
        return []
    with open(cam_anns_path) as f:
        all_anns = json.load(f)
    return all_anns.get(camera_id, [])


def main():
    img_bgr = cv2.imread(IMAGE_PATH)
    assert img_bgr is not None, f"Could not read {IMAGE_PATH}"
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    gt_anns = _load_gt_annotations(CAM_ANNS, CAMERA_ID)
    if gt_anns:
        print(f"GT annotations loaded: {[a['point_id'] for a in gt_anns]}")
    else:
        print(f"No GT annotations found for {CAMERA_ID}.")

    detected = []   # list of (x, y) clicked by the user

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        f"{CAMERA_ID}  |  left-click: add detected   right-click: remove   "
        f"[c] clear   [s / close]: save & exit",
        fontsize=10,
    )
    ax.axis("off")
    ax.imshow(img_rgb)

    # ── Static GT annotation markers ────────────────────────────────────────
    for ann in gt_anns:
        ax.plot(ann["x"], ann["y"], marker="D", markersize=10,
                color="lime", markeredgecolor="black", markeredgewidth=1.2,
                zorder=4)
        ax.text(ann["x"] + 14, ann["y"] - 10, ann["point_id"],
                color="lime", fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5, lw=0),
                zorder=4)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor="lime",
               markeredgecolor="black", markersize=9, label="GT annotation"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="yellow",
               markeredgecolor="black", markersize=9, label="Detected (clicked)"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=9,
              framealpha=0.7, facecolor="black", labelcolor="white")

    # ── Interactive detected markers ─────────────────────────────────────────
    scatter = ax.scatter([], [], s=150, c="yellow", edgecolors="black",
                         linewidths=1.5, zorder=5)
    det_label_artists = []

    def _refresh():
        xs = [p[0] for p in detected]
        ys = [p[1] for p in detected]
        scatter.set_offsets(list(zip(xs, ys)) if detected else np.empty((0, 2)))

        for t in det_label_artists:
            t.remove()
        det_label_artists.clear()

        for i, (px, py) in enumerate(detected):
            t = ax.text(px + 14, py - 10, f"D{i+1}",
                        color="yellow", fontsize=9, fontweight="bold",
                        bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5, lw=0),
                        zorder=5)
            det_label_artists.append(t)

        fig.canvas.draw_idle()

    def on_click(event):
        if event.inaxes != ax or event.xdata is None:
            return
        x, y = event.xdata, event.ydata
        if event.button == 1:
            detected.append((x, y))
            print(f"  Added D{len(detected)}  ({x:.1f}, {y:.1f})")
        elif event.button == 3:
            idx = _nearest_idx(detected, x, y)
            if idx >= 0:
                removed = detected.pop(idx)
                print(f"  Removed point at ({removed[0]:.1f}, {removed[1]:.1f})")
        _refresh()

    def on_key(event):
        if event.key == "c":
            detected.clear()
            print("  Cleared all detected points.")
            _refresh()
        elif event.key in ("s", "q"):
            _save_and_close()

    def on_close(_event):
        _save_and_close(reopen=False)

    def _save_and_close(reopen=True):
        out = img_bgr.copy()

        # Draw GT annotations (green diamonds → drawn as circles with a cross inside)
        for ann in gt_anns:
            u, v = int(round(ann["x"])), int(round(ann["y"]))
            cv2.circle(out, (u, v), GT_RADIUS, _GT_COLOR, -1, cv2.LINE_AA)
            cv2.circle(out, (u, v), GT_RADIUS, (0, 0, 0),  2, cv2.LINE_AA)
            # Small cross to distinguish from detected dots
            cv2.line(out, (u - GT_RADIUS, v), (u + GT_RADIUS, v), (0, 0, 0), 2, cv2.LINE_AA)
            cv2.line(out, (u, v - GT_RADIUS), (u, v + GT_RADIUS), (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(out, ann["point_id"], (u + 14, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),   3, cv2.LINE_AA)
            cv2.putText(out, ann["point_id"], (u + 14, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, _GT_COLOR,   1, cv2.LINE_AA)

        # Draw detected (clicked) markers
        for i, (px, py) in enumerate(detected):
            u, v = int(round(px)), int(round(py))
            cv2.circle(out, (u, v), MARKER_RADIUS, _DET_COLOR, -1, cv2.LINE_AA)
            cv2.circle(out, (u, v), MARKER_RADIUS, (0, 0, 0),   2, cv2.LINE_AA)
            label = f"D{i+1}"
            cv2.putText(out, label, (u + 14, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0),    3, cv2.LINE_AA)
            cv2.putText(out, label, (u + 14, v - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, _LABEL_WHITE,  1, cv2.LINE_AA)

        cv2.imwrite(OUTPUT_PATH, out)
        print(f"\nSaved → {OUTPUT_PATH}")
        print(f"  GT points : {[a['point_id'] for a in gt_anns]}")
        print(f"  Detected  : {len(detected)} point(s)")
        for i, (px, py) in enumerate(detected):
            print(f"    D{i+1}: ({px:.1f}, {py:.1f})")

        if reopen:
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("key_press_event",    on_key)
    fig.canvas.mpl_connect("close_event",        on_close)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
