import cv2
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import os

def generate_wall_instance_visualization(img_path, depth_path, mask_path, K_matrix, output_path):
    # 1. Load the data
    img = cv2.imread(img_path)
    raw_depth = np.load(depth_path)
    
    # Extract red wall mask (using the color thresholds from earlier)
    hsv = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2HSV)
    wall_mask = cv2.bitwise_or(
        cv2.inRange(hsv, np.array([0, 50, 50]), np.array([10, 255, 255])),
        cv2.inRange(hsv, np.array([170, 50, 50]), np.array([180, 255, 255]))
    )
    
    # Match resolutions
    if wall_mask.shape[:2] != raw_depth.shape[:2]:
        wall_mask = cv2.resize(wall_mask, (raw_depth.shape[1], raw_depth.shape[0]), interpolation=cv2.INTER_NEAREST)

    # 2. Extract EXACT pixel coordinates (u, v) and their depth (z)
    v, u = np.where(wall_mask == 255)
    z = raw_depth[v, u]
    
    # Filter out invalid depth pixels
    valid = z > 0.0
    u, v, z = u[valid], v[valid], z[valid]

    # 3. Manual 2D-to-3D Unprojection (Preserving index order!)
    fx, fy = K_matrix[0, 0], K_matrix[1, 1]
    cx, cy = K_matrix[0, 2], K_matrix[1, 2]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    points_3d = np.column_stack((x, y, z)).astype(np.float64)

    # 4. Open3D Normal Estimation
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_3d)
    
    print("Estimating normals for all wall pixels... (This may take a few seconds without downsampling)")
    # We use a larger radius here to smooth over the "curved" depth network noise
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=50))
    pcd.orient_normals_towards_camera_location(np.array([0., 0., 0.]))
    
    normals = np.asarray(pcd.normals)

    # 5. Cluster the Normals
    print("Clustering walls...")
    clustering = DBSCAN(eps=0.15, min_samples=500).fit(normals)
    labels = clustering.labels_

    # 6. Paint the 2D Image!
    # Create an empty overlay image
    h_img, w_img = raw_depth.shape
    overlay = np.zeros((h_img, w_img, 3), dtype=np.uint8)
    
    # Define some distinct, bright colors for different walls (BGR format)
    colors = [
        [0, 255, 0],   # Green (Wall 1)
        [255, 0, 0],   # Blue (Wall 2)
        [0, 255, 255], # Yellow (Wall 3)
        [255, 0, 255], # Magenta (Wall 4)
        [0, 165, 255]  # Orange (Wall 5)
    ]
    noise_color = [128, 128, 128] # Grey for curved/noisy patches

    # Map the labels back to the pixels
    unique_labels = set(labels)
    for k in unique_labels:
        # Find which of our flattened (u,v) arrays belong to this cluster
        cluster_indices = (labels == k)
        cluster_u = u[cluster_indices]
        cluster_v = v[cluster_indices]
        
        if k == -1:
            # Noise pixels
            overlay[cluster_v, cluster_u] = noise_color
        else:
            # Valid Wall pixels
            color = colors[k % len(colors)]
            overlay[cluster_v, cluster_u] = color

    # 7. Blend with the original image
    # Resize img if it doesn't match depth resolution
    if img.shape[:2] != (h_img, w_img):
        img = cv2.resize(img, (w_img, h_img))
        
    # Darken the original image slightly so the bright walls pop out
    darkened_img = cv2.convertScaleAbs(img, alpha=0.5, beta=0)
    
    # Create a mask of where we painted
    painted_mask = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY) > 0
    
    # Combine them
    final_output = darkened_img.copy()
    final_output[painted_mask] = overlay[painted_mask]

    # Save
    cv2.imwrite(output_path, final_output)
    print(f"Visualization saved to {output_path}")


# --- Execution ---
if __name__ == "__main__":
    
    # Replace these with actual paths from your dataset
    IMG_PATH = "/home/user/thesis/code/dataset/Point_Detection_Tests/Camera_10_frame.jpg"
    DEPTH_PATH = "/home/user/thesis/code/depth/temporal_depth/Camera_10_temporal_depth_raw.npy"
    MASK_PATH = "/home/user/thesis/code/segmentation/temporal_masks/Camera_10_temporal_bg.jpg"
    
    # Replace with Camera 10's K matrix
    K_MATRIX = np.array([
        [1662.7, 0, 960], 
        [0, 1662.7, 540], 
        [0, 0, 1]
    ])
    
    OUTPUT_VIS = "/home/user/thesis/code/geometry/Camera_10_wall_instances.jpg"
    
    generate_wall_instance_visualization(IMG_PATH, DEPTH_PATH, MASK_PATH, K_MATRIX, OUTPUT_VIS)