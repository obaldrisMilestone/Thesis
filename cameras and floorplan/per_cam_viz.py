import json
import math
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import ipywidgets as widgets
from IPython.display import display

# --- 1. CONFIGURATION ---
JSON_FILE_PATH = '/home/user/thesis/code/dataset/Hospital_000_Val/MTMC_Tracking_2025/val/Hospital_000/calibration.json'  # Path to your JSON file 
MAP_IMAGE_PATH = '/home/user/thesis/code/dataset/Hospital_000_Val/MTMC_Tracking_2025/val/Hospital_000/map.png'      
CAMERA_IMAGES_DIR = '/home/user/thesis/code/dataset/Point_Detection_Tests/' # Folder containing your camera frames
VIEW_DISTANCE = 500             

# --- 3. JSON & IMAGE LOADING HELPERS ---
def load_camera_data(json_path):
    try:
        with open(json_path, 'r') as file:
            data = json.load(file)
        sensors = data.get('sensors', [])
        return [sensor for sensor in sensors if sensor.get('type') == 'camera']
    except Exception as e:
        print(f"Error loading JSON: {e}")
        return []

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
            yaw_deg = float(attr['value'].split(',')[0])
            break

    YAW_OFFSET_DEG = -90  
    # Invert the angle because image Y coordinates go down, not up
    yaw_rad = math.radians(yaw_deg + YAW_OFFSET_DEG)
            
    
    fx = cam_json['intrinsicMatrix'][0][0]
    img_width = 0
    for attr in cam_json.get('attributes', []):
        if attr['name'] == 'frameWidth':
            img_width = float(attr['value'])
            break
            
    hfov_rad = 2 * math.atan(img_width / (2 * fx))
    return Point(pixel_x, pixel_y), yaw_rad, hfov_rad



# ... [Keep Sections 1 through 6 exactly the same as before] ...
import glob # Make sure this is imported at the top!

# --- 7. NATIVE INTERACTIVE VIEWER (FOR LOCAL EXECUTION) ---
class LocalCameraViewer:
    def __init__(self, camera_list, map_img_path):
        self.cameras = camera_list
        self.current_idx = 0
        self.map_img = mpimg.imread(map_img_path)
        self.img_h, self.img_w = self.map_img.shape[:2]
        
        # Setup Figure
        self.fig, (self.ax_map, self.ax_cam) = plt.subplots(1, 2, figsize=(18, 8))
        self.fig.canvas.manager.set_window_title('Camera FoV Viewer - Use LEFT/RIGHT Arrows')
        
        # Bind keyboard events
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.draw_current()
        plt.show()

    def on_key(self, event):
        """Listens for left/right arrow keys to change cameras."""
        if event.key == 'right':
            self.current_idx = (self.current_idx + 1) % len(self.cameras)
            self.draw_current()
        elif event.key == 'left':
            self.current_idx = (self.current_idx - 1) % len(self.cameras)
            self.draw_current()

    def draw_current(self):
        """Draws the map and camera image."""
        self.ax_map.clear()
        self.ax_cam.clear()
        
        cam_json = self.cameras[self.current_idx]
        cam_id = cam_json.get('id', 'Unknown')
        
        # --- LEFT: MAP & FOV ---
        self.ax_map.imshow(self.map_img)
        cam_pt, yaw, hfov = get_camera_fov_params(cam_json, 0, self.img_h)
        
   
        left_angle = yaw - (hfov / 2)
        right_angle = yaw + (hfov / 2)
        lx = cam_pt.x + VIEW_DISTANCE * math.cos(left_angle)
        ly = cam_pt.y + VIEW_DISTANCE * math.sin(left_angle)
        rx = cam_pt.x + VIEW_DISTANCE * math.cos(right_angle)
        ry = cam_pt.y + VIEW_DISTANCE * math.sin(right_angle)
        
        self.ax_map.plot([cam_pt.x, lx], [cam_pt.y, ly], color='blue', linestyle='--', alpha=0.7)
        self.ax_map.plot([cam_pt.x, rx], [cam_pt.y, ry], color='blue', linestyle='--', alpha=0.7)
        self.ax_map.fill([cam_pt.x, lx, rx], [cam_pt.y, ly, ry], color='blue', alpha=0.1)
        self.ax_map.scatter([cam_pt.x], [cam_pt.y], color='red', s=100, zorder=5)
        
        self.ax_map.set_title(f"Map View: {cam_id} ({self.current_idx+1}/{len(self.cameras)})\n[Press Left/Right Arrows to switch]")
        self.ax_map.axis('off')

        # Look for any file that contains the cam_id
        search_pattern = os.path.join(CAMERA_IMAGES_DIR, f"*{cam_id}*")
        all_found_files = glob.glob(search_pattern)
        
        # Filter OUT non-image files (like your metadata.json files)
        valid_extensions = ('.png', '.jpg', '.jpeg')
        found_images = [f for f in all_found_files if f.lower().endswith(valid_extensions)]
        
        if found_images:
            matched_image_path = found_images[0]
            self.ax_cam.imshow(mpimg.imread(matched_image_path))
            self.ax_cam.set_title(f"Camera View: {os.path.basename(matched_image_path)}")
        else:
            self.ax_cam.text(0.5, 0.5, f"Image not found for\nCamera ID: '{cam_id}'", 
                        horizontalalignment='center', verticalalignment='center', 
                        fontsize=12, color='red')
            self.ax_cam.set_facecolor('black')
            self.ax_cam.set_title(f"Camera View: {cam_id}")
            
        self.ax_cam.axis('off')
        self.fig.canvas.draw_idle()

# --- RUN SCRIPT ---
if __name__ == "__main__":
    cameras = load_camera_data(JSON_FILE_PATH)
    if cameras:
        print(f"Loaded {len(cameras)} cameras. Opening viewer... Use LEFT and RIGHT arrow keys to navigate.")
        viewer = LocalCameraViewer(cameras, MAP_IMAGE_PATH)
    else:
        print("No cameras found.")