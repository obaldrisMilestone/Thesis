import numpy as np
import open3d as o3d
import cv2

class FloorNormalExtractor:
    def __init__(self, K_matrix):
        """
        Initializes the extractor with the camera's Intrinsic matrix (K).
        """
        self.K = K_matrix

    def extract_normal(self, raw_depth_map, floor_mask):
        """
        Takes the raw floating-point depth map (.npy) and the binary floor mask (.jpg),
        and calculates the 3D surface normal of the floor.
        """
        # 1. Ensure the mask and depth map are the exact same resolution
        if floor_mask.shape[:2] != raw_depth_map.shape[:2]:
            floor_mask = cv2.resize(
                floor_mask, 
                (raw_depth_map.shape[1], raw_depth_map.shape[0]), 
                interpolation=cv2.INTER_NEAREST
            )

        # 2. Mask the Depth Map
        # We only keep depth values where the mask is 255 (Floor). Everything else becomes 0.0.
        masked_depth = np.where(floor_mask == 255, raw_depth_map, 0.0).astype(np.float32)

        # 3. Convert to Open3D Format
        depth_image = o3d.geometry.Image(masked_depth)

        h, w = raw_depth_map.shape
        fx, fy = self.K[0, 0], self.K[1, 1]
        cx, cy = self.K[0, 2], self.K[1, 2]
        
        intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, fx, fy, cx, cy)

        # 4. Generate Point Cloud (Open3D automatically ignores depth=0 pixels)
        pcd = o3d.geometry.PointCloud.create_from_depth_image(depth_image, intrinsics)
        
        # Downsample for significantly faster processing
        pcd = pcd.voxel_down_sample(voxel_size=0.05)
        
        if len(pcd.points) < 100:
            print("Warning: Not enough floor points found in the masked area.")
            return None, None

        # 5. Fit a mathematical plane using RANSAC
        # plane_model returns [A, B, C, D] from the equation Ax + By + Cz + D = 0
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=0.05,
            ransac_n=3,
            num_iterations=1000
        )
        
        a, b, c, d = plane_model
        normal = np.array([a, b, c])

        # 6. Directional Normalization
        # In OpenCV camera coordinates, the Y-axis points DOWN into the floor.
        # We want our floor normal pointing UP towards the camera, so the Y component should be negative.
        if normal[1] > 0:
            normal = -normal
            d = -d

        camera_height = abs(d)

        print(f"Floor Normal Vector (Nx, Ny, Nz): [{normal[0]:.4f}, {normal[1]:.4f}, {normal[2]:.4f}]")
        print(f"Estimated Camera Height: {camera_height:.3f} units")

        return normal, camera_height

# --- Example Usage ---
if __name__ == "__main__":
    # Example Camera Intrinsic Matrix (Replace with your actual calibration)
    K_MATRIX = np.array([
        [800.0, 0.0,   640.0],
        [0.0,   800.0, 360.0],
        [0.0,   0.0,   1.0]
    ])
    
    extractor = FloorNormalExtractor(K_MATRIX)
    
    # Load your files (e.g., from the outputs of your previous scripts)
    # Using the raw .npy depth file preserves the exact float distances!
    depth_array = np.load("./dataset/temporal_depth/Camera_18_temporal_depth_raw.npy") 
    floor_binary_mask = cv2.imread("./dataset/temporal_masks/Camera_18_temporal_bg.jpg", cv2.IMREAD_GRAYSCALE)
    
    # We threshold just in case JPEG compression made the binary mask fuzzy
    _, floor_binary_mask = cv2.threshold(floor_binary_mask, 127, 255, cv2.THRESH_BINARY)
    
    normal, height = extractor.extract_normal(depth_array, floor_binary_mask)