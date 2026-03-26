import os
import glob
import json
import cv2
import numpy as np

def create_bev_homography(K, normal, height, bev_res_px_per_m=50, bev_size=(800, 800)):
    """
    Creates a Homography matrix that warps a perspective image into a top-down BEV.
    """
    n = np.array(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    
    if n[1] > 0:
        n = -n

    cam_forward = np.array([0.0, 0.0, 1.0])
    Y_floor = cam_forward - (np.dot(cam_forward, n) * n)
    Y_floor = Y_floor / np.linalg.norm(Y_floor)
    
    X_floor = np.cross(Y_floor, n)
    X_floor = X_floor / np.linalg.norm(X_floor)

    # Expanded 16x16 meter canvas
    X_min, X_max = -8.0, 8.0
    Y_min, Y_max = 0.0, 16.0 
    
    local_metric_pts = [
        (X_min, Y_min), 
        (X_max, Y_min), 
        (X_min, Y_max), 
        (X_max, Y_max)  
    ]

    src_pixels = []
    dst_pixels = []

    W, H_img = bev_size
    U_center = W // 2
    V_camera = H_img 

    for X, Y in local_metric_pts:
        P_cam = X * X_floor + Y * Y_floor - height * n
        
        p_img = K @ P_cam
        u = p_img[0] / p_img[2]
        v = p_img[1] / p_img[2]
        src_pixels.append([u, v])
        
        U_bev = U_center + (X * bev_res_px_per_m)
        V_bev = V_camera - (Y * bev_res_px_per_m) 
        dst_pixels.append([U_bev, V_bev])

    src_pixels = np.array(src_pixels, dtype=np.float32)
    dst_pixels = np.array(dst_pixels, dtype=np.float32)
    
    H = cv2.getPerspectiveTransform(src_pixels, dst_pixels)
    return H

# --- Main Batch Execution ---
if __name__ == "__main__":
    
    # --- Configuration Paths ---
    PREDICTIONS_JSON = "/home/user/thesis/code/geometry/camera_extrinsics.json"
    META_DIR = "/home/user/thesis/code/dataset/Point_Detection_Tests"
    IMG_DIR = "/home/user/thesis/code/dataset/Point_Detection_Tests" # Assuming frames are here
    MASK_DIR = "/home/user/thesis/code/segmentation/temporal_masks"
    
    OUTPUT_DIR = "/home/user/thesis/code/geometry/bev_projections"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # BEV Settings
    BEV_PX_PER_M = 15
    BEV_WIDTH = 2000  # 16 meters
    BEV_HEIGHT = 2000 # 16 meters

    # 1. Load the predicted normals
    if not os.path.exists(PREDICTIONS_JSON):
        raise FileNotFoundError(f"Could not find extrinsics at {PREDICTIONS_JSON}")
        
    with open(PREDICTIONS_JSON, 'r') as f:
        extrinsics_data = json.load(f)

    print(f"Starting batch BEV projection for {len(extrinsics_data)} cameras...\n")

    # 2. Iterate through each camera we successfully processed earlier
    for camera_id, pred_data in extrinsics_data.items():
        print(f"Processing {camera_id}...")
        
        try:
            # Construct expected file paths
            meta_path = os.path.join(META_DIR, f"{camera_id}_metadata.json")
            img_path = os.path.join(IMG_DIR, f"{camera_id}_frame.jpg")
            mask_path = os.path.join(MASK_DIR, f"{camera_id}_temporal_bg.jpg")
            
            # Verify required files exist
            if not os.path.exists(meta_path):
                print(f"  -> Missing metadata, skipping.")
                continue
            if not os.path.exists(img_path):
                print(f"  -> Missing RGB frame, skipping.")
                continue
            if not os.path.exists(mask_path):
                print(f"  -> Missing Mask, skipping.")
                continue

            # Load Data
            predicted_normal = pred_data["floor_normal"]
            img = cv2.imread(img_path)
            mask = cv2.imread(mask_path)
            
            # Extract K matrix and True Height from Metadata
            with open(meta_path, 'r') as f:
                meta = json.load(f)
                
            K_matrix = np.array(meta["calibration"]["intrinsicMatrix"])
            ext_matrix = np.array(meta["calibration"]["extrinsicMatrix"])
            R = ext_matrix[:, :3]
            t = ext_matrix[:, 3]
            gt_camera_center = -np.dot(R.T, t)
            true_height = abs(gt_camera_center[2])

            # Generate Homography
            H = create_bev_homography(
                K_matrix, 
                predicted_normal, 
                true_height, 
                bev_res_px_per_m=BEV_PX_PER_M, 
                bev_size=(BEV_WIDTH, BEV_HEIGHT)
            )

            # Warp Images
            bev_img = cv2.warpPerspective(img, H, (BEV_WIDTH, BEV_HEIGHT))
            
            # Using INTER_NEAREST for the mask is critical to prevent class colors from blending
            bev_mask = cv2.warpPerspective(mask, H, (BEV_WIDTH, BEV_HEIGHT), flags=cv2.INTER_NEAREST)

            # Save Results (Save mask as PNG to prevent JPEG artifacting on sharp edges)
            out_img_path = os.path.join(OUTPUT_DIR, f"{camera_id}_bev_rgb.jpg")
            out_mask_path = os.path.join(OUTPUT_DIR, f"{camera_id}_bev_mask.png")
            
            cv2.imwrite(out_img_path, bev_img)
            cv2.imwrite(out_mask_path, bev_mask)
            
            print(f"  -> Success! Saved BEV to {out_img_path}")

        except Exception as e:
            print(f"  -> Error processing {camera_id}: {e}")

    print("\nBatch BEV generation complete! Check your output directory.")