import os
import glob
import json
import numpy as np
import open3d as o3d
import cv2

class MetricFloorNormalExtractor:
    def __init__(self):
        pass

    def parse_intrinsics(self, json_path):
        """
        Reads the camera calibration JSON and extracts the 3x3 Intrinsic Matrix (K).
        Adapts to standard calibration dictionary structures.
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Drill down into the calibration dictionary 
        # (Based on the structure built in AI-SmartSpaces_processing.py)
        calib = data.get('calibration', {})
        
        # Datasets often store the intrinsic matrix under 'K', 'intrinsics', or 'intrinsic_matrix'
        if 'intrinsicMatrix' in calib:
            return np.array(calib['intrinsicMatrix'])
        elif 'intrinsics' in calib:
            return np.array(calib['intrinsics'])
        elif 'intrinsic_matrix' in calib:
            return np.array(calib['intrinsic_matrix'])
        else:
            raise ValueError(f"Could not find a valid K matrix in {json_path}")

    def extract_normal(self, raw_depth_map, floor_mask, K_matrix):
        """
        Masks the depth map, unprojects it to 3D using K, and finds the floor plane.
        """
        # 1. Match resolutions if they differ
        if floor_mask.shape[:2] != raw_depth_map.shape[:2]:
            floor_mask = cv2.resize(
                floor_mask, 
                (raw_depth_map.shape[1], raw_depth_map.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )

        # 2. Mask the depth (keep only floor pixels, zero out the rest)
        masked_depth = np.where(floor_mask == 255, raw_depth_map, 0.0).astype(np.float32)

        # 3. Manual unprojection — avoids depth_scale assumptions in Open3D
        fx, fy = K_matrix[0, 0], K_matrix[1, 1]
        cx, cy = K_matrix[0, 2], K_matrix[1, 2]

        v_idx, u_idx = np.where(masked_depth > 0)
        z = masked_depth[v_idx, u_idx]
        x = (u_idx - cx) * z / fx
        y = (v_idx - cy) * z / fy

        if len(z) < 100:
            print("  -> Warning: Not enough 3D points to confidently fit a plane.")
            return None, None

        points = np.vstack([x, y, z]).T
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        # 4. Adaptive thresholds — scale with median depth so metric and
        #    relative depth maps both work
        median_z = float(np.median(z))
        voxel_size        = median_z * 0.01   # 1% of median depth
        ransac_threshold  = median_z * 0.01   # 1% of median depth

        pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

        if len(pcd.points) < 100:
            print("  -> Warning: Not enough 3D points to confidently fit a plane.")
            return None, None

        # 5. Fit Mathematical Plane (RANSAC)
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=ransac_threshold,
            ransac_n=3,
            num_iterations=1000
        )
        
        a, b, c, d = plane_model
        normal = np.array([a, b, c])

        # 7. Directional Normalization
        # OpenCV camera axes: X is right, Y is DOWN, Z is forward.
        # We want the floor normal pointing UP towards the ceiling, so the Y component must be negative.
        if normal[1] > 0:
            normal = -normal
            d = -d

        camera_height = abs(d)
        return normal, camera_height

# --- Main Batch Execution ---
if __name__ == "__main__":
    
    # Define where the different pieces of data live
    DEPTH_DIR = "/home/user/thesis/code/depth/temporal_depth"
    MASK_DIR = "/home/user/thesis/code/segmentation/temporal_masks"
    META_DIR = "/home/user/thesis/code/dataset/Point_Detection_Tests" # From your AI-SmartSpaces_processing.py script
    
    # Output file
    OUTPUT_JSON_PATH = "./geometry/camera_extrinsics.json"
    
    extractor = MetricFloorNormalExtractor()
    metadata_files = glob.glob(os.path.join(META_DIR, "*_metadata.json"))
    
    # Dictionary to hold all the results before saving
    all_camera_results = {}
    
    if not metadata_files:
        print(f"No metadata JSONs found in '{META_DIR}'.")
        
    for meta_path in metadata_files:
        filename = os.path.basename(meta_path)
        camera_id = filename.replace("_metadata.json", "")
        print(f"\nProcessing {camera_id}...")
        
        try:
            expected_depth_name = f"{camera_id}_temporal_depth_raw.npy"
            expected_mask_name = f"{camera_id}_temporal_bg.jpg"
            
            depth_path = os.path.join(DEPTH_DIR, expected_depth_name)
            mask_path = os.path.join(MASK_DIR, expected_mask_name)
            
            if not os.path.exists(depth_path):
                print(f"  -> Skipping: Missing depth map at {depth_path}")
                continue
            if not os.path.exists(mask_path):
                print(f"  -> Skipping: Missing mask at {mask_path}")
                continue
                
            K_matrix = extractor.parse_intrinsics(meta_path)
            depth_array = np.load(depth_path)
            
            floor_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, floor_mask = cv2.threshold(floor_mask, 127, 255, cv2.THRESH_BINARY)
            
            normal, height = extractor.extract_normal(depth_array, floor_mask, K_matrix)
            
            if normal is not None:
                print(f"  -> Metric Floor Normal: [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")
                print(f"  -> Unscaled Camera Height: {height:.3f} units")
                
                # Store the results in the dictionary. 
                # We must cast numpy floats to native Python floats for json.dump() to work!
                all_camera_results[camera_id] = {
                    "floor_normal": [float(normal[0]), float(normal[1]), float(normal[2])],
                    "camera_height": float(height)
                }
                
        except Exception as e:
            print(f"  -> Error processing {camera_id}: {e}")
            
    # Save the accumulated results to a master JSON file
    if all_camera_results:
        with open(OUTPUT_JSON_PATH, 'w') as f:
            json.dump(all_camera_results, f, indent=4)
        print(f"\nSuccessfully saved all extrinsics data to {OUTPUT_JSON_PATH}")
    else:
        print("\nNo results were extracted to save.")