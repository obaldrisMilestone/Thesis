import json
import math
import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# --- 1. CONFIGURATION ---
JSON_FILE_PATH = '/home/user/thesis/code/dataset/Hospital_000_Val/MTMC_Tracking_2025/val/Hospital_000/calibration.json'
MAP_IMAGE_PATH = '/home/user/thesis/code/dataset/Hospital_000_Val/MTMC_Tracking_2025/val/Hospital_000/map.png'      
CAMERA_IMAGES_DIR = '/home/user/thesis/code/dataset/Point_Detection_Tests/' 
VIEW_DISTANCE = 500             

# Output files for the annotations
OUT_MAP_JSON = 'annotated_floorplan.json'
OUT_CAM_JSON = 'annotated_cameras.json'

# --- 2. MOCK CAD MAP WALLS ---
CAD_WALLS = [
    [400, 200, 400, 800],  
    [400, 200, 800, 200],  
    [500, 400, 600, 400],  
    [800, 200, 800, 800]
]

# --- 3. JSON LOADING HELPERS ---
def load_camera_data(json_path):
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        sensors = data.get('sensors', [])
        return [sensor for sensor in sensors if sensor.get('type') == 'camera']
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return []

def load_annotations():
    """Loads existing annotations if they exist, otherwise returns empty structures."""
    map_data, cam_data = {}, {}
    global_id = 0
    
    if os.path.exists(OUT_MAP_JSON):
        with open(OUT_MAP_JSON, 'r') as f:
            map_data = json.load(f)
            if map_data:
                # Find the highest ID to continue counting from there
                ids = [int(k.replace('P_', '')) for k in map_data.keys()]
                global_id = max(ids) + 1 if ids else 0
                
    if os.path.exists(OUT_CAM_JSON):
        with open(OUT_CAM_JSON, 'r') as f:
            cam_data = json.load(f)
            
    return map_data, cam_data, global_id

def save_annotations(map_data, cam_data):
    with open(OUT_MAP_JSON, 'w') as f:
        json.dump(map_data, f, indent=4)
    with open(OUT_CAM_JSON, 'w') as f:
        json.dump(cam_data, f, indent=4)

# --- 4. GEOMETRY HELPERS ---
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

# --- 5. CAMERA PROCESSING ---
def get_camera_fov_params(cam_json, center_x, center_y):
    local_x, local_y = cam_json['coordinates']['x'], cam_json['coordinates']['y']
    scale = cam_json['scaleFactor']
    trans_x, trans_y = cam_json['translationToGlobalCoordinates']['x'], cam_json['translationToGlobalCoordinates']['y']
    
    pixel_x = center_x + ((local_x * scale) + trans_x*scale)
    pixel_y = center_y - ((local_y * scale) + trans_y*scale) 
    
    yaw_deg = 0.0
    for attr in cam_json.get('attributes', []):
        if attr['name'] == 'direction':
            yaw_deg = float(attr['value'])
            break
            
    # Your updated Yaw Logic
    YAW_OFFSET_DEG = -90  
    yaw_rad = math.radians(yaw_deg + YAW_OFFSET_DEG)
    
    fx = cam_json['intrinsicMatrix'][0][0]
    img_width = 0
    for attr in cam_json.get('attributes', []):
        if attr['name'] == 'frameWidth':
            img_width = float(attr['value'])
            break
            
    hfov_rad = 2 * math.atan(img_width / (2 * fx))
    return Point(pixel_x, pixel_y), yaw_rad, hfov_rad

def find_visible_walls(cam_pt, yaw, hfov, all_walls):
    visible_walls = []
    for wall in all_walls:
        p1 = Point(wall[0], wall[1])
        p2 = Point(wall[2], wall[3])
        angle_p1 = math.atan2(p1.y - cam_pt.y, p1.x - cam_pt.x)
        angle_p2 = math.atan2(p2.y - cam_pt.y, p2.x - cam_pt.x)
        
        if abs(get_angle_diff(angle_p1, yaw)) > hfov / 2 or \
           abs(get_angle_diff(angle_p2, yaw)) > hfov / 2:
            continue 
            
        is_occluded = False
        for other_wall in all_walls:
            if wall == other_wall: continue 
            w1 = Point(other_wall[0], other_wall[1])
            w2 = Point(other_wall[2], other_wall[3])
            if segments_intersect(cam_pt, p1, w1, w2) or segments_intersect(cam_pt, p2, w1, w2):
                is_occluded = True
                break
        if not is_occluded:
            visible_walls.append(wall)
    return visible_walls

# --- 6. ANNOTATION VIEWER CLASS ---
class CameraAnnotator:
    def __init__(self, camera_list, map_img_path):
        self.cameras = camera_list
        self.current_idx = 0
        self.map_img = mpimg.imread(map_img_path)
        self.img_h, self.img_w = self.map_img.shape[:2]
        
        # Load existing Data
        self.map_data, self.cam_data, self.global_pt_id = load_annotations()
        
        # State machine variables for clicking
        self.pending_map_pt = None
        self.pending_cam_pt = None
        
        # Setup Figure
        self.fig, (self.ax_map, self.ax_cam) = plt.subplots(1, 2, figsize=(18, 8))
        self.fig.canvas.manager.set_window_title('Point Annotator - Left Click to mark, Left/Right Arrows to switch')
        
        # Bind events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        self.draw_current()
        plt.show()

    def on_key(self, event):
        """Handle keyboard navigation."""
        if event.key == 'right':
            self.current_idx = (self.current_idx + 1) % len(self.cameras)
            self.pending_map_pt = None
            self.pending_cam_pt = None
            self.draw_current()
        elif event.key == 'left':
            self.current_idx = (self.current_idx - 1) % len(self.cameras)
            self.pending_map_pt = None
            self.pending_cam_pt = None
            self.draw_current()
        elif event.key == 'escape':
            # Clear pending selection
            self.pending_map_pt = None
            self.pending_cam_pt = None
            self.draw_current()

    def on_click(self, event):
        """Handle mouse clicks for annotation."""
        # Ensure click is inside one of the axes and is a left click (button 1)
        if event.inaxes is None or event.button != 1:
            return

        if event.inaxes == self.ax_map:
            self.pending_map_pt = (event.xdata, event.ydata)
        elif event.inaxes == self.ax_cam:
            self.pending_cam_pt = (event.xdata, event.ydata)

        # Check if we have a complete pair
        if self.pending_map_pt is not None and self.pending_cam_pt is not None:
            self.save_pair()
            
        self.draw_current()

    def save_pair(self):
        """Saves the completed pair to the dictionaries and JSON files."""
        pt_id_str = f"P_{self.global_pt_id}"
        cam_id = self.cameras[self.current_idx].get('id', 'Unknown')
        
        # Save to Map Dictionary
        self.map_data[pt_id_str] = {
            "x": self.pending_map_pt[0],
            "y": self.pending_map_pt[1]
        }
        
        # Save to Camera Dictionary
        if cam_id not in self.cam_data:
            self.cam_data[cam_id] = []
            
        self.cam_data[cam_id].append({
            "point_id": pt_id_str,
            "x": self.pending_cam_pt[0],
            "y": self.pending_cam_pt[1]
        })
        
        # Increment global ID and clear pending
        self.global_pt_id += 1
        self.pending_map_pt = None
        self.pending_cam_pt = None
        
        # Write to disk
        save_annotations(self.map_data, self.cam_data)
        print(f"Saved pair {pt_id_str} to JSON files.")

    def draw_current(self):
        """Draws the map, camera, FoV, and all points."""
        self.ax_map.clear()
        self.ax_cam.clear()
        
        cam_json = self.cameras[self.current_idx]
        cam_id = cam_json.get('id', 'Unknown')
        
        # ==========================================
        # LEFT: MAP & FOV
        # ==========================================
        self.ax_map.imshow(self.map_img)
        cam_pt, yaw, hfov = get_camera_fov_params(cam_json, 0, self.img_h)
        visible_walls = find_visible_walls(cam_pt, yaw, hfov, CAD_WALLS)
        
        for w in CAD_WALLS:
            self.ax_map.plot([w[0], w[2]], [w[1], w[3]], color='gray', linewidth=3, alpha=0.5)
        for w in visible_walls:
            self.ax_map.plot([w[0], w[2]], [w[1], w[3]], color='lime', linewidth=5)
            
        left_angle = yaw - (hfov / 2)
        right_angle = yaw + (hfov / 2)
        lx = cam_pt.x + VIEW_DISTANCE * math.cos(left_angle)
        ly = cam_pt.y + VIEW_DISTANCE * math.sin(left_angle)
        rx = cam_pt.x + VIEW_DISTANCE * math.cos(right_angle)
        ry = cam_pt.y + VIEW_DISTANCE * math.sin(right_angle)
        
        self.ax_map.plot([cam_pt.x, lx], [cam_pt.y, ly], color='blue', linestyle='--', alpha=0.7)
        self.ax_map.plot([cam_pt.x, rx], [cam_pt.y, ry], color='blue', linestyle='--', alpha=0.7)
        self.ax_map.fill([cam_pt.x, lx, rx], [cam_pt.y, ly, ry], color='blue', alpha=0.1)
        self.ax_map.scatter([cam_pt.x], [cam_pt.y], color='blue', s=100, marker='^', zorder=5) # Cam Position
        
        # Draw ALL saved Map Points
        for pt_id, coords in self.map_data.items():
            self.ax_map.scatter(coords['x'], coords['y'], color='yellow', edgecolor='black', s=80, zorder=6)
            self.ax_map.text(coords['x'] + 10, coords['y'] - 10, pt_id, color='yellow', fontsize=10, weight='bold', zorder=7)

        # Draw PENDING Map Point
        if self.pending_map_pt:
            self.ax_map.scatter(self.pending_map_pt[0], self.pending_map_pt[1], color='red', s=100, zorder=8)
            self.ax_map.text(self.pending_map_pt[0] + 10, self.pending_map_pt[1] - 10, "PENDING", color='red')

        self.ax_map.set_title(f"Map View: {cam_id} ({self.current_idx+1}/{len(self.cameras)})\n"
                              f"[Arrows] Switch Cam  |  [ESC] Clear Selection")
        self.ax_map.axis('off')

        # ==========================================
        # RIGHT: ACTUAL CAMERA IMAGE
        # ==========================================
        search_pattern = os.path.join(CAMERA_IMAGES_DIR, f"*{cam_id}*")
        all_found_files = glob.glob(search_pattern)
        valid_extensions = ('.png', '.jpg', '.jpeg')
        found_images = [f for f in all_found_files if f.lower().endswith(valid_extensions)]
        
        if found_images:
            matched_image_path = found_images[0]
            self.ax_cam.imshow(mpimg.imread(matched_image_path))
            self.ax_cam.set_title(f"Camera: {cam_id}")
            
            # Draw saved Camera Points for THIS specific camera
            if cam_id in self.cam_data:
                for pt in self.cam_data[cam_id]:
                    self.ax_cam.scatter(pt['x'], pt['y'], color='yellow', edgecolor='black', s=80, zorder=6)
                    self.ax_cam.text(pt['x'] + 10, pt['y'] - 10, pt['point_id'], color='yellow', fontsize=10, weight='bold', zorder=7)
            
            # Draw PENDING Camera Point
            if self.pending_cam_pt:
                self.ax_cam.scatter(self.pending_cam_pt[0], self.pending_cam_pt[1], color='red', s=100, zorder=8)
                self.ax_cam.text(self.pending_cam_pt[0] + 10, self.pending_cam_pt[1] - 10, "PENDING", color='red')
                
        else:
            self.ax_cam.text(0.5, 0.5, f"Image not found for\nCamera ID: '{cam_id}'", 
                        horizontalalignment='center', verticalalignment='center', 
                        fontsize=12, color='red')
            self.ax_cam.set_facecolor('black')
            self.ax_cam.set_title(f"Camera View: {cam_id}")
            
        self.ax_cam.axis('off')
        
        # --- UI STATE TEXT ---
        state_text = "READY: Click Map or Camera"
        if self.pending_map_pt and not self.pending_cam_pt:
            state_text = "MAP MARKED: Now click the Camera image."
        elif self.pending_cam_pt and not self.pending_map_pt:
            state_text = "CAMERA MARKED: Now click the Map image."
            
        self.fig.suptitle(f"Annotation Tool | Status: {state_text}", fontsize=14, weight='bold', color='purple')
        
        self.fig.canvas.draw_idle()

# --- 7. RUN SCRIPT ---
if __name__ == "__main__":
    cameras = load_camera_data(JSON_FILE_PATH)
    if cameras:
        print(f"Loaded {len(cameras)} cameras. Opening Annotator...")
        print("INSTRUCTIONS:")
        print(" 1. Click on the Map to mark a floorplan coordinate.")
        print(" 2. Click on the Camera image to mark the matching pixel coordinate.")
        print(" 3. The pair is saved automatically once both are clicked.")
        print(" 4. Use LEFT and RIGHT arrows to change cameras. Press ESC to cancel a pending click.")
        annotator = CameraAnnotator(cameras, MAP_IMAGE_PATH)
    else:
        print("No cameras found.")