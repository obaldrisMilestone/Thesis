"""
Annotation Visualizer
---------------------
Displays annotated floorplan ↔ camera correspondences produced by annotator_v2.py.

Config: set SCENARIO_DIR to a dataset scenario, or set ANNOTATIONS_DIR + CAMERA_IMAGES_DIR
directly for a custom layout.
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches

# --- CONFIGURATION ---
# Option A: point at a scenario directory (same as annotator_v2.py)
SCENARIO_DIR = '/home/user/thesis/code/dataset/MTMC_Tracking_2025/train/Warehouse_000'

# Option B: override with explicit paths (leave None to auto-detect from SCENARIO_DIR)
ANNOTATIONS_DIR = None   # e.g. 'point_matching' for the old hospital annotations
CAMERA_IMAGES_DIR = None # e.g. '/home/user/thesis/code/dataset/Point_Detection_Tests'
MAP_IMAGE_PATH = None

# ---- auto-detect from SCENARIO_DIR ----
_DATASET_MARKER = 'MTMC_Tracking_2025'
if ANNOTATIONS_DIR is None:
    _parts = SCENARIO_DIR.split(_DATASET_MARKER)
    _scenario_name = _parts[-1].strip('/')
    ANNOTATIONS_DIR = os.path.join(os.path.dirname(__file__), 'annotations', _scenario_name)

if CAMERA_IMAGES_DIR is None:
    CAMERA_IMAGES_DIR = os.path.join(SCENARIO_DIR, 'depth_maps')

if MAP_IMAGE_PATH is None:
    MAP_IMAGE_PATH = os.path.join(SCENARIO_DIR, 'map.png')

# ---- annotation file paths ----
FLOORPLAN_JSON = os.path.join(ANNOTATIONS_DIR, 'annotated_floorplan.json')
CAMERAS_JSON   = os.path.join(ANNOTATIONS_DIR, 'annotated_cameras.json')
SKIPPED_JSON   = os.path.join(ANNOTATIONS_DIR, 'skipped_cameras.json')

# ---- color palette — up to 20 distinct cameras ----
_PALETTE = [
    '#e6194b', '#3cb44b', '#4363d8', '#f58231', '#911eb4',
    '#42d4f4', '#f032e6', '#bfef45', '#fabed4', '#469990',
    '#dcbeff', '#9A6324', '#fffac8', '#800000', '#aaffc3',
    '#808000', '#ffd8b1', '#000075', '#a9a9a9', '#ffffff',
]

# ---- helpers ----

def load_json(path):
    if not os.path.exists(path):
        return {}
    with open(path, 'r') as f:
        return json.load(f)

def find_camera_image(cam_id, images_dir):
    direct = os.path.join(images_dir, f"{cam_id}_bg.png")
    if os.path.exists(direct):
        return direct
    matches = glob.glob(os.path.join(images_dir, f"*{cam_id}*"))
    valid = [f for f in matches if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return valid[0] if valid else None

def color_for_index(i):
    return _PALETTE[i % len(_PALETTE)]


# ---- main visualizer class ----

class AnnotationVisualizer:
    def __init__(self):
        self.map_img = mpimg.imread(MAP_IMAGE_PATH)

        self.floorplan = load_json(FLOORPLAN_JSON)   # {pt_id: {x, y}}
        self.cameras   = load_json(CAMERAS_JSON)     # {cam_id: [{point_id, x, y}]}
        self.skipped   = set(load_json(SKIPPED_JSON)) if os.path.exists(SKIPPED_JSON) else set()

        if not self.cameras:
            raise RuntimeError(f"No annotation data found in {CAMERAS_JSON}")

        self.cam_ids = sorted(self.cameras.keys())
        self.cam_color = {cam_id: color_for_index(i) for i, cam_id in enumerate(self.cam_ids)}
        self.current_idx = 0

        # Build reverse lookup: point_id → floorplan (x, y)
        self.fp_lookup = self.floorplan  # already keyed by point_id

        self._setup_figure()
        self.draw()
        plt.show()

    def _setup_figure(self):
        self.fig = plt.figure(figsize=(22, 10))
        self.fig.patch.set_facecolor('#1a1a2e')

        # Left panel: floorplan (wider)
        self.ax_map = self.fig.add_axes([0.01, 0.08, 0.52, 0.86])
        # Right panel: camera image
        self.ax_cam = self.fig.add_axes([0.55, 0.08, 0.44, 0.86])

        self.fig.canvas.manager.set_window_title('Annotation Visualizer')
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_key(self, event):
        if event.key == 'right':
            self.current_idx = (self.current_idx + 1) % len(self.cam_ids)
            self.draw()
        elif event.key == 'left':
            self.current_idx = (self.current_idx - 1) % len(self.cam_ids)
            self.draw()

    # ---- drawing ----

    def draw(self):
        self.ax_map.clear()
        self.ax_cam.clear()

        cur_cam_id = self.cam_ids[self.current_idx]
        cur_color  = self.cam_color[cur_cam_id]
        cur_pts    = self.cameras[cur_cam_id]  # [{point_id, x, y}]

        self._draw_floorplan(cur_cam_id, cur_color, cur_pts)
        self._draw_camera(cur_cam_id, cur_color, cur_pts)
        self._draw_legend()
        self._draw_title()

        self.fig.canvas.draw_idle()

    def _draw_floorplan(self, cur_cam_id, cur_color, cur_pts):
        ax = self.ax_map
        ax.imshow(self.map_img, alpha=0.85)

        # Draw all other cameras' points — faint
        for cam_id, pts in self.cameras.items():
            if cam_id == cur_cam_id:
                continue
            color = self.cam_color[cam_id]
            for pt in pts:
                fp = self.fp_lookup.get(pt['point_id'])
                if fp is None:
                    continue
                ax.scatter(fp['x'], fp['y'], color=color, s=30, alpha=0.3,
                           edgecolors='black', linewidths=0.5, zorder=4)

        # Draw current camera's points — prominent
        for pt in cur_pts:
            fp = self.fp_lookup.get(pt['point_id'])
            if fp is None:
                continue
            ax.scatter(fp['x'], fp['y'], color=cur_color, s=120,
                       edgecolors='white', linewidths=1.2, zorder=6)
            ax.text(fp['x'] + 10, fp['y'] - 10, pt['point_id'],
                    color='white', fontsize=8, weight='bold', zorder=7,
                    bbox=dict(boxstyle='round,pad=0.2', fc=cur_color, alpha=0.7, ec='none'))

        ax.set_title(f"Floorplan — {len(self.fp_lookup)} total points across {len(self.cam_ids)} cameras",
                     color='white', fontsize=11, pad=6)
        ax.axis('off')
        ax.set_facecolor('#1a1a2e')

    def _draw_camera(self, cur_cam_id, cur_color, cur_pts):
        ax = self.ax_cam
        img_path = find_camera_image(cur_cam_id, CAMERA_IMAGES_DIR)

        if img_path:
            ax.imshow(mpimg.imread(img_path))
        else:
            ax.set_facecolor('#111')
            ax.text(0.5, 0.5, f"Image not found\n{cur_cam_id}",
                    ha='center', va='center', color='red', fontsize=13,
                    transform=ax.transAxes)

        for pt in cur_pts:
            x, y = pt['x'], pt['y']
            # Outer ring for visibility on any background
            ax.scatter(x, y, color='white', s=220, zorder=6, linewidths=0)
            ax.scatter(x, y, color=cur_color, s=130,
                       edgecolors='black', linewidths=0.8, zorder=7)
            ax.text(x + 12, y - 12, pt['point_id'],
                    color='white', fontsize=9, weight='bold', zorder=8,
                    bbox=dict(boxstyle='round,pad=0.2', fc=cur_color, alpha=0.85, ec='none'))

        n = len(cur_pts)
        ax.set_title(f"{cur_cam_id}  —  {n} annotated point{'s' if n != 1 else ''}",
                     color='white', fontsize=12, pad=6)
        ax.axis('off')
        ax.set_facecolor('#1a1a2e')

    def _draw_legend(self):
        patches = []
        for cam_id in self.cam_ids:
            n = len(self.cameras[cam_id])
            label = f"{cam_id}  ({n}pt)"
            patches.append(mpatches.Patch(color=self.cam_color[cam_id], label=label))

        self.fig.legend(
            handles=patches,
            loc='lower center',
            ncol=min(len(patches), 8),
            fontsize=7.5,
            framealpha=0.25,
            facecolor='#1a1a2e',
            edgecolor='gray',
            labelcolor='white',
            bbox_to_anchor=(0.5, 0.0),
        )

    def _draw_title(self):
        total_pts = sum(len(v) for v in self.cameras.values())
        skipped_info = f"  |  {len(self.skipped)} skipped" if self.skipped else ""
        self.fig.suptitle(
            f"[←/→] navigate cameras    "
            f"{self.current_idx + 1} / {len(self.cam_ids)}    "
            f"total points: {total_pts}{skipped_info}",
            fontsize=11, color='#aaaaaa', y=0.99,
        )


# ---- entry point ----

if __name__ == "__main__":
    missing = []
    for label, path in [('map', MAP_IMAGE_PATH), ('floorplan JSON', FLOORPLAN_JSON), ('cameras JSON', CAMERAS_JSON)]:
        if not os.path.exists(path):
            missing.append(f"  {label}: {path}")

    if missing:
        print("Missing files:")
        for m in missing:
            print(m)
    else:
        print(f"Annotations : {ANNOTATIONS_DIR}")
        print(f"Map         : {MAP_IMAGE_PATH}")
        print(f"Camera imgs : {CAMERA_IMAGES_DIR}")
        AnnotationVisualizer()