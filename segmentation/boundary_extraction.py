import os
import glob
import cv2
import numpy as np
import json

import cv2
import numpy as np
import json
import math

def merge_collinear_lines(lines, angle_tolerance_deg=5, distance_tolerance_px=30):
    """
    Groups parallel and collinear line segments and merges them into single, 
    continuous mathematical line segments spanning the maximum endpoints.
    """
    if lines is None or len(lines) == 0:
        return []

    # 1. Convert lines to Angle (theta) and Perpendicular Distance (rho)
    line_data = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        if x1 == x2 and y1 == y2: continue
        
        # Calculate angle in radians [0, pi)
        theta = math.atan2(y2 - y1, x2 - x1)
        if theta < 0:
            theta += np.pi
            
        # Calculate line normal (nx, ny)
        nx = -math.sin(theta)
        ny = math.cos(theta)
        
        # Calculate perpendicular distance from origin (rho)
        rho = x1 * nx + y1 * ny
        if rho < 0:
            rho = -rho
            theta -= np.pi
            if theta < 0: theta += np.pi

        line_data.append({
            'pts': [x1, y1, x2, y2],
            'theta': theta,
            'rho': rho
        })

    # 2. Group lines that share similar theta and rho
    angle_tol_rad = math.radians(angle_tolerance_deg)
    groups = []
    
    for ld in line_data:
        matched = False
        for group in groups:
            g_theta = group['theta']
            g_rho = group['rho']
            
            # Angle difference (handling pi wrap-around)
            d_theta = min(abs(ld['theta'] - g_theta), np.pi - abs(ld['theta'] - g_theta))
            d_rho = abs(ld['rho'] - g_rho)
            
            if d_theta < angle_tol_rad and d_rho < distance_tolerance_px:
                group['lines'].append(ld['pts'])
                matched = True
                break
                
        if not matched:
            groups.append({
                'theta': ld['theta'],
                'rho': ld['rho'],
                'lines': [ld['pts']]
            })

    # 3. Fit a single continuous line through each group
    merged_lines = []
    for group in groups:
        # Collect all X,Y endpoints in this group
        points = []
        for pts in group['lines']:
            points.append((pts[0], pts[1]))
            points.append((pts[2], pts[3]))
        points = np.array(points, dtype=np.float32)
        
        # Fit the best mathematical line through all endpoints using Least Squares
        [vx, vy, x0, y0] = cv2.fitLine(points, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]
        
        # Project all points onto this new vector to find the extreme start and end points
        projections = []
        for p in points:
            proj = (p[0] - x0) * vx + (p[1] - y0) * vy
            projections.append(proj)
            
        min_proj = min(projections)
        max_proj = max(projections)
        
        # Calculate the final merged coordinates
        x_start = int(round(x0 + min_proj * vx))
        y_start = int(round(y0 + min_proj * vy))
        x_end = int(round(x0 + max_proj * vx))
        y_end = int(round(y0 + max_proj * vy))
        
        merged_lines.append([x_start, y_start, x_end, y_end])

    return merged_lines

def batch_extract_straight_wall_bases(input_dir, output_dir, output_json):
    """
    Processes a directory of temporal mask visualizations, extracts straight 
    wall-to-floor boundary lines, saves visualization images, and exports 
    the line coordinates to a JSON file for downstream optimization.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all the temporal background images
    image_paths = glob.glob(os.path.join(input_dir, "*_temporal_bg.jpg"))
    
    if not image_paths:
        print(f"No temporal background images found in {input_dir}")
        return

    # Dictionary to hold the line coordinates for all cameras
    all_extracted_lines = {}

    print(f"Starting batch boundary extraction for {len(image_paths)} cameras...\n")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        camera_id = filename.replace("_temporal_bg.jpg", "")
        print(f"Processing {camera_id}...")

        # 1. Load the visualization image
        img = cv2.imread(img_path)
        if img is None:
            print(f"  -> Error: Could not load {filename}")
            continue
            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 2. Isolate the Semantic Classes using HSV Color Thresholding
        # Extract Green (Floor)
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])
        floor_mask = cv2.inRange(hsv, lower_green, upper_green)

        # Extract Red (Wall)
        lower_red1 = np.array([0, 50, 50])
        upper_red1 = np.array([10, 255, 255])
        lower_red2 = np.array([170, 50, 50])
        upper_red2 = np.array([180, 255, 255])
        wall_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )

        # 3. Find the Strict Intersection (Wall Base)
        # Dilate the floor by a 5x5 pixel radius
        kernel = np.ones((5, 5), np.uint8)
        floor_dilated = cv2.dilate(floor_mask, kernel, iterations=1)
        
        # Keep only the dilated floor pixels that land exactly on a wall pixel
        boundary_pixels = cv2.bitwise_and(floor_dilated, wall_mask)

# ... (Previous code: Masking and Dilating) ...

        # 4. Fit Straight Lines using Probabilistic Hough Transform
        raw_lines = cv2.HoughLinesP(
            boundary_pixels,
            rho=1,
            theta=np.pi / 180,
            threshold=50, 
            minLineLength=80, 
            maxLineGap=50
        )

        # --- NEW STEP: Merge the fragmented lines ---
        merged_lines = merge_collinear_lines(
            raw_lines, 
            angle_tolerance_deg=5,     # Allow up to 5 degrees of wiggle room
            distance_tolerance_px=30   # Allow lines up to 30 pixels parallel to merge
        )

        # 5. Process and Draw Results
        output_img = img.copy()
        camera_lines = []

        for line in merged_lines:
            x1, y1, x2, y2 = line
            
            # Cast to native Python int for JSON serialization!
            camera_lines.append([int(x1), int(y1), int(x2), int(y2)])
            
            # Draw a thick blue line over the merged wall bases
            cv2.line(output_img, (x1, y1), (x2, y2), (255, 0, 0), 3)
                
        all_extracted_lines[camera_id] = camera_lines
        print(f"  -> Extracted and merged into {len(camera_lines)} continuous wall lines.")

        # ... (Rest of the saving logic) ...

        # 6. Save the output images locally
        boundary_save_path = os.path.join(output_dir, f"{camera_id}_boundary_raw.jpg")
        lines_save_path = os.path.join(output_dir, f"{camera_id}_boundary_lines.jpg")
        
        cv2.imwrite(boundary_save_path, boundary_pixels)
        cv2.imwrite(lines_save_path, output_img)

    # 7. Export all coordinates to a master JSON file
    if all_extracted_lines:
        with open(output_json, 'w') as f:
            json.dump(all_extracted_lines, f, indent=4)
        print(f"\nSuccessfully saved all 2D line coordinates to {output_json}")


# --- Execution ---
if __name__ == "__main__":
    
    # Define your specific workspace paths
    INPUT_MASKS_DIR = "/home/user/thesis/code/segmentation/temporal_masks"
    OUTPUT_LINES_DIR = "/home/user/thesis/code/segmentation/boundary_lines_colinear"
    OUTPUT_JSON_PATH = "/home/user/thesis/code/geometry/extracted_2d_lines.json"
    
    batch_extract_straight_wall_bases(INPUT_MASKS_DIR, OUTPUT_LINES_DIR, OUTPUT_JSON_PATH)