import json
import math
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- 1. CONFIGURATION ---
SCENARIO_DIR = '/home/user/thesis/code/dataset/MTMC_Tracking_2025/val/Hospital_000'

# Derived paths (auto-detected from SCENARIO_DIR)
JSON_FILE_PATH = os.path.join(SCENARIO_DIR, 'calibration.json')
MAP_IMAGE_PATH = os.path.join(SCENARIO_DIR, 'map.png')
IMAGES_DIR = os.path.join(SCENARIO_DIR, 'images')

# Output: annotations/<split>/<scenario>/ relative to this script
_DATASET_MARKER = 'MTMC_Tracking_2025'
_parts = SCENARIO_DIR.split(_DATASET_MARKER)
SCENARIO_NAME = _parts[-1].strip('/')  # e.g. "train/Warehouse_000"
ANNOTATIONS_DIR = os.path.join(os.path.dirname(__file__), 'annotations', SCENARIO_NAME)

VIEW_DISTANCE = 500

# --- 2. JSON LOADING HELPERS ---
def load_camera_data(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        return [s for s in data.get('sensors', []) if s.get('type') == 'camera']
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return []

def load_annotations(annotations_dir):
    map_json = os.path.join(annotations_dir, 'annotated_floorplan.json')
    cam_json = os.path.join(annotations_dir, 'annotated_cameras.json')
    skip_json = os.path.join(annotations_dir, 'skipped_cameras.json')

    map_data, cam_data, skipped = {}, {}, set()
    global_id = 0

    if os.path.exists(map_json):
        with open(map_json, 'r') as f:
            map_data = json.load(f)
        if map_data:
            ids = [int(k.replace('P_', '')) for k in map_data.keys()]
            global_id = max(ids) + 1 if ids else 0

    if os.path.exists(cam_json):
        with open(cam_json, 'r') as f:
            cam_data = json.load(f)

    if os.path.exists(skip_json):
        with open(skip_json, 'r') as f:
            skipped = set(json.load(f))

    return map_data, cam_data, skipped, global_id

def save_annotations(annotations_dir, map_data, cam_data, skipped):
    os.makedirs(annotations_dir, exist_ok=True)
    with open(os.path.join(annotations_dir, 'annotated_floorplan.json'), 'w') as f:
        json.dump(map_data, f, indent=4)
    with open(os.path.join(annotations_dir, 'annotated_cameras.json'), 'w') as f:
        json.dump(cam_data, f, indent=4)
    with open(os.path.join(annotations_dir, 'skipped_cameras.json'), 'w') as f:
        json.dump(sorted(skipped), f, indent=4)

# --- 3. GEOMETRY HELPERS ---
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

def ccw(A, B, C):
    return (C.y - A.y) * (B.x - A.x) > (B.y - A.y) * (C.x - A.x)

def segments_intersect(A, B, C, D):
    if (A.x == C.x and A.y == C.y) or (A.x == D.x and A.y == D.y) or \
       (B.x == C.x and B.y == C.y) or (B.x == D.x and B.y == D.y):
        return False
    return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

def get_angle_diff(angle1, angle2):
    return (angle1 - angle2 + math.pi) % (2 * math.pi) - math.pi

# --- 4. CAMERA FOV HELPERS ---
def get_camera_fov_params(cam_json, img_h):
    local_x = cam_json['coordinates']['x']
    local_y = cam_json['coordinates']['y']
    scale = cam_json['scaleFactor']
    trans_x = cam_json['translationToGlobalCoordinates']['x']
    trans_y = cam_json['translationToGlobalCoordinates']['y']

    pixel_x = (local_x * scale) + trans_x * scale
    pixel_y = img_h - ((local_y * scale) + trans_y * scale)

    yaw_deg = 0.0
    img_width = 0.0
    for attr in cam_json.get('attributes', []):
        if attr['name'] == 'direction':
            yaw_deg = float(attr['value'])
        elif attr['name'] == 'frameWidth':
            img_width = float(attr['value'])

    YAW_OFFSET_DEG = -90
    yaw_rad = math.radians(yaw_deg + YAW_OFFSET_DEG)
    fx = cam_json['intrinsicMatrix'][0][0]
    hfov_rad = 2 * math.atan(img_width / (2 * fx)) if fx > 0 else math.radians(90)

    return Point(pixel_x, pixel_y), yaw_rad, hfov_rad

def find_camera_image(cam_id, images_dir):
    for ext in ('.jpg', '.jpeg', '.png'):
        p = os.path.join(images_dir, f"{cam_id}{ext}")
        if os.path.exists(p):
            return p
    # Fallback: first image whose stem starts with cam_id
    matches = glob.glob(os.path.join(images_dir, f"{cam_id}*"))
    valid = [f for f in matches if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    return valid[0] if valid else None

# --- 5. ANNOTATOR CLASS ---
class CameraAnnotatorV2:
    def __init__(self, camera_list, map_img_path, images_dir, annotations_dir):
        self.cameras = camera_list
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.current_idx = 0

        self.map_img = mpimg.imread(map_img_path)
        self.img_h, self.img_w = self.map_img.shape[:2]

        self.map_data, self.cam_data, self.skipped, self.global_pt_id = \
            load_annotations(annotations_dir)

        self.pending_map_pt = None
        self.pending_cam_pt = None

        self.fig, (self.ax_map, self.ax_cam) = plt.subplots(1, 2, figsize=(20, 9))
        self.fig.canvas.manager.set_window_title(
            'Annotator v2 — Arrows: switch | S: skip/unskip | ESC: clear'
        )

        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.draw_current()
        plt.show()

    # ---- state helpers ----

    def current_cam_id(self):
        return self.cameras[self.current_idx].get('id', f'cam_{self.current_idx}')

    def is_skipped(self):
        return self.current_cam_id() in self.skipped

    # ---- event handlers ----

    def on_key(self, event):
        if event.key == 'right':
            self.current_idx = (self.current_idx + 1) % len(self.cameras)
            self._clear_pending()
        elif event.key == 'left':
            self.current_idx = (self.current_idx - 1) % len(self.cameras)
            self._clear_pending()
        elif event.key == 's':
            self._toggle_skip()
        elif event.key == 'escape':
            self._clear_pending()
        self.draw_current()

    def on_click(self, event):
        if event.inaxes is None or event.button != 1:
            return
        if self.is_skipped():
            return

        if event.inaxes == self.ax_map:
            self.pending_map_pt = (event.xdata, event.ydata)
        elif event.inaxes == self.ax_cam:
            self.pending_cam_pt = (event.xdata, event.ydata)

        if self.pending_map_pt is not None and self.pending_cam_pt is not None:
            self._save_pair()

        self.draw_current()

    # ---- actions ----

    def _clear_pending(self):
        self.pending_map_pt = None
        self.pending_cam_pt = None

    def _toggle_skip(self):
        cam_id = self.current_cam_id()
        if cam_id in self.skipped:
            self.skipped.discard(cam_id)
            print(f"Unskipped {cam_id}.")
        else:
            self.skipped.add(cam_id)
            self._clear_pending()
            print(f"Skipped {cam_id} — will not appear in annotations.")
        save_annotations(self.annotations_dir, self.map_data, self.cam_data, self.skipped)

    def _save_pair(self):
        pt_id_str = f"P_{self.global_pt_id}"
        cam_id = self.current_cam_id()

        self.map_data[pt_id_str] = {
            "x": self.pending_map_pt[0],
            "y": self.pending_map_pt[1]
        }

        if cam_id not in self.cam_data:
            self.cam_data[cam_id] = []
        self.cam_data[cam_id].append({
            "point_id": pt_id_str,
            "x": self.pending_cam_pt[0],
            "y": self.pending_cam_pt[1]
        })

        self.global_pt_id += 1
        self._clear_pending()
        save_annotations(self.annotations_dir, self.map_data, self.cam_data, self.skipped)
        print(f"Saved {pt_id_str} for {cam_id}.")

    # ---- drawing ----

    def draw_current(self):
        self.ax_map.clear()
        self.ax_cam.clear()

        cam_json = self.cameras[self.current_idx]
        cam_id = self.current_cam_id()
        skipped = self.is_skipped()

        # ---- LEFT: MAP ----
        self.ax_map.imshow(self.map_img)

        try:
            cam_pt, yaw, hfov = get_camera_fov_params(cam_json, self.img_h)
            left_angle = yaw - hfov / 2
            right_angle = yaw + hfov / 2
            lx = cam_pt.x + VIEW_DISTANCE * math.cos(left_angle)
            ly = cam_pt.y + VIEW_DISTANCE * math.sin(left_angle)
            rx = cam_pt.x + VIEW_DISTANCE * math.cos(right_angle)
            ry = cam_pt.y + VIEW_DISTANCE * math.sin(right_angle)

            fov_color = 'gray' if skipped else 'blue'
            self.ax_map.plot([cam_pt.x, lx], [cam_pt.y, ly], color=fov_color, linestyle='--', alpha=0.6)
            self.ax_map.plot([cam_pt.x, rx], [cam_pt.y, ry], color=fov_color, linestyle='--', alpha=0.6)
            self.ax_map.fill([cam_pt.x, lx, rx], [cam_pt.y, ly, ry], color=fov_color, alpha=0.1)
            marker_color = 'gray' if skipped else 'blue'
            self.ax_map.scatter([cam_pt.x], [cam_pt.y], color=marker_color, s=120, marker='^', zorder=5)
        except Exception:
            pass

        # All saved map points
        for pt_id, coords in self.map_data.items():
            self.ax_map.scatter(coords['x'], coords['y'], color='yellow', edgecolors='black', s=80, zorder=6)
            self.ax_map.text(coords['x'] + 8, coords['y'] - 8, pt_id, color='yellow',
                             fontsize=8, weight='bold', zorder=7)

        if self.pending_map_pt:
            self.ax_map.scatter(*self.pending_map_pt, color='red', s=120, zorder=8)
            self.ax_map.text(self.pending_map_pt[0] + 8, self.pending_map_pt[1] - 8,
                             "PENDING", color='red', fontsize=8)

        skip_suffix = "  [SKIPPED — press S to unskip]" if skipped else ""
        self.ax_map.set_title(
            f"Map — {cam_id}  ({self.current_idx + 1}/{len(self.cameras)}){skip_suffix}\n"
            f"[←/→] Switch  |  [S] Skip/Unskip  |  [ESC] Clear"
        )
        self.ax_map.axis('off')

        # ---- RIGHT: CAMERA IMAGE ----
        img_path = find_camera_image(cam_id, self.images_dir)

        if img_path:
            self.ax_cam.imshow(mpimg.imread(img_path))

            if cam_id in self.cam_data:
                for pt in self.cam_data[cam_id]:
                    self.ax_cam.scatter(pt['x'], pt['y'], color='yellow', edgecolors='black', s=80, zorder=6)
                    self.ax_cam.text(pt['x'] + 8, pt['y'] - 8, pt['point_id'],
                                     color='yellow', fontsize=8, weight='bold', zorder=7)

            if self.pending_cam_pt:
                self.ax_cam.scatter(*self.pending_cam_pt, color='red', s=120, zorder=8)
                self.ax_cam.text(self.pending_cam_pt[0] + 8, self.pending_cam_pt[1] - 8,
                                 "PENDING", color='red', fontsize=8)
        else:
            self.ax_cam.text(0.5, 0.5, f"No image found for\n{cam_id}",
                             ha='center', va='center', fontsize=12, color='red')
            self.ax_cam.set_facecolor('black')

        cam_pts_count = len(self.cam_data.get(cam_id, []))
        self.ax_cam.set_title(f"Camera: {cam_id}  |  {cam_pts_count} point(s) annotated")
        self.ax_cam.axis('off')

        # ---- STATUS BAR ----
        if skipped:
            state = "SKIPPED — press S to unskip this camera"
            color = 'gray'
        elif self.pending_map_pt and not self.pending_cam_pt:
            state = "MAP MARKED — now click the camera image"
            color = 'darkorange'
        elif self.pending_cam_pt and not self.pending_map_pt:
            state = "CAMERA MARKED — now click the map"
            color = 'darkorange'
        else:
            state = "READY — click map or camera to start a pair"
            color = 'purple'

        self.fig.suptitle(f"Annotator v2  |  {SCENARIO_NAME}  |  {state}",
                          fontsize=13, weight='bold', color=color)
        self.fig.canvas.draw_idle()


# --- 6. RUN ---
if __name__ == "__main__":
    cameras = load_camera_data(JSON_FILE_PATH)
    if not cameras:
        print("No cameras found in calibration.json.")
    else:
        os.makedirs(ANNOTATIONS_DIR, exist_ok=True)
        print(f"Scenario : {SCENARIO_NAME}")
        print(f"Cameras  : {len(cameras)}")
        print(f"Saving to: {ANNOTATIONS_DIR}")
        print()
        print("Controls:")
        print("  Left / Right  — switch camera")
        print("  S             — skip / unskip current camera (excluded from annotated_cameras.json)")
        print("  ESC           — cancel pending click")
        print("  Left-click    — mark a point on the map, then on the camera (pair saved automatically)")
        CameraAnnotatorV2(cameras, MAP_IMAGE_PATH, IMAGES_DIR, ANNOTATIONS_DIR)
