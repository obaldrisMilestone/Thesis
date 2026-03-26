import os
import glob
import json
import cv2
import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt

class WallNormalExtractor:
    def __init__(self, K_matrix):
        self.K = K_matrix

    def extract_wall_mask(self, img_path):
        """Isolates the Red 'Wall' class from the temporal background image."""
        img = cv2.imread(img_path)
        if img is None:
            return None
            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 50, 50]), np.array([180, 255, 255])
        
        wall_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )
        return wall_mask

    def process_camera(self, depth_path, mask_path):
        """Converts depth to 3D, extracts normals, and clusters them."""
        # 1. Load Data
        raw_depth = np.load(depth_path)
        wall_mask = self.extract_wall_mask(mask_path)
        if wall_mask is None: return None, None, None

        # Resize mask if necessary
        if wall_mask.shape[:2] != raw_depth.shape[:2]:
            wall_mask = cv2.resize(wall_mask, (raw_depth.shape[1], raw_depth.shape[0]), interpolation=cv2.INTER_NEAREST)

        # 2. Mask Depth and Convert to Open3D
        masked_depth = np.where(wall_mask == 255, raw_depth, 0.0).astype(np.float32)
        depth_image = o3d.geometry.Image(masked_depth)

        h, w = raw_depth.shape
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        # 3. Create Point Cloud
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, intrinsics)
        pcd = pcd.voxel_down_sample(voxel_size=0.1) # Downsample aggressively for speed and noise reduction
        
        if len(pcd.points) < 100:
            return None, None, None

        # 4. Estimate Normals
        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.5, max_nn=30)
        )
        
        # CRITICAL: Force all normals to point towards the camera origin (0,0,0)
        pcd.orient_normals_towards_camera_location(np.array([0.0, 0.0, 0.0]))
        
        normals = np.asarray(pcd.normals)

        # 5. Cluster Normals on the Gaussian Sphere using DBSCAN
        # eps=0.15 corresponds roughly to an 8.5-degree tolerance between normal vectors
        clustering = DBSCAN(eps=0.05, min_samples=200).fit(normals)
        labels = clustering.labels_

        # 6. Extract Dominant Clusters
        unique_labels = set(labels)
        dominant_normals = []
        
        for k in unique_labels:
            if k == -1: # DBSCAN marks noise as -1
                continue
                
            # Get all normal vectors in this cluster
            cluster_normals = normals[labels == k]
            
            # Calculate the centroid (average direction) of the cluster
            centroid = np.mean(cluster_normals, axis=0)
            
            # Re-normalize to ensure length is exactly 1.0
            centroid = centroid / np.linalg.norm(centroid)
            dominant_normals.append(centroid.tolist())

        return dominant_normals, normals, labels

    def plot_gaussian_sphere(self, normals, labels, dominant_normals, save_path):
        """Visualizes the clustered normals on a 3D sphere."""
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot noise as small grey dots
        noise_mask = (labels == -1)
        ax.scatter(normals[noise_mask, 0], normals[noise_mask, 1], normals[noise_mask, 2], 
                   c='gray', s=1, alpha=0.1, label='Noise')

        # Plot valid clusters
        unique_labels = set(labels) - {-1}
        cmap = plt.get_cmap('tab10')
        
        for i, k in enumerate(unique_labels):
            cluster_mask = (labels == k)
            ax.scatter(normals[cluster_mask, 0], normals[cluster_mask, 1], normals[cluster_mask, 2], 
                       c=[cmap(i)], s=5, alpha=0.5, label=f'Wall {i+1}')

        # Plot the calculated centroids as large red stars
        for centroid in dominant_normals:
            ax.scatter(centroid[0], centroid[1], centroid[2], c='red', s=200, marker='*', edgecolor='black')

        # Format plot
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        ax.set_title("Gaussian Sphere: Wall Normal Clustering")
        plt.legend()
        plt.savefig(save_path)
        plt.close()


# --- Main Batch Execution ---
if __name__ == "__main__":
    
    DEPTH_DIR = "/home/user/thesis/code/depth/temporal_depth"
    MASK_DIR = "/home/user/thesis/code/segmentation/temporal_masks"
    META_DIR = "/home/user/thesis/code/dataset/Point_Detection_Tests"
    OUTPUT_DIR = "/home/user/thesis/code/geometry/walls_normals"
    OUTPUT_JSON = os.path.join(OUTPUT_DIR, "dominant_wall_normals.json")
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    metadata_files = glob.glob(os.path.join(META_DIR, "*_metadata.json"))
    all_camera_normals = {}

    print(f"Starting 3D Normal Extraction for {len(metadata_files)} cameras...\n")

    for meta_path in metadata_files:
        camera_id = os.path.basename(meta_path).replace("_metadata.json", "")
        print(f"Processing {camera_id}...")
        
        try:
            depth_path = os.path.join(DEPTH_DIR, f"{camera_id}_temporal_depth_raw.npy")
            mask_path = os.path.join(MASK_DIR, f"{camera_id}_temporal_bg.jpg")
            
            if not os.path.exists(depth_path) or not os.path.exists(mask_path):
                print(f"  -> Missing files, skipping.")
                continue

            with open(meta_path, 'r') as f:
                meta = json.load(f)
            K_matrix = np.array(meta["calibration"]["intrinsicMatrix"])

            extractor = WallNormalExtractor(K_matrix)
            dominant_normals, raw_normals, labels = extractor.process_camera(depth_path, mask_path)
            
            if dominant_normals:
                print(f"  -> Found {len(dominant_normals)} dominant wall direction(s).")
                all_camera_normals[camera_id] = dominant_normals
                
                # Save visualization
                plot_path = os.path.join(OUTPUT_DIR, f"{camera_id}_gaussian_sphere.png")
                extractor.plot_gaussian_sphere(raw_normals, labels, dominant_normals, plot_path)
            else:
                print(f"  -> Warning: No strong normal clusters found.")

        except Exception as e:
            print(f"  -> Error: {e}")

    # Export to JSON
    if all_camera_normals:
        with open(OUTPUT_JSON, 'w') as f:
            json.dump(all_camera_normals, f, indent=4)
        print(f"\nSuccessfully saved dominant normals to {OUTPUT_JSON}")