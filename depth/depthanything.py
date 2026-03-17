import os
import glob
import torch
import cv2
import numpy as np
import open3d as o3d
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class DepthAnythingModule:
    def __init__(self, model_name="LiheYoung/depth-anything-small-hf"):
        """
        Loads the Depth Anything model. 
        You can use 'small-hf', 'base-hf', or 'large-hf' depending on your VRAM.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading DepthAnything to {self.device}...")
        
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def get_depth_map(self, image_path):
        image = Image.open(image_path).convert("RGB")
        
        # Prepare image for the model
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predicted_depth = outputs.predicted_depth
            
        # Interpolate to original resolution
        predicted_depth = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=image.size[::-1],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
        
        # Convert to numpy. 
        # Note: Depth Anything outputs relative inverse depth (disparity).
        # To make it act like true depth for a point cloud, we invert it.
        depth_map = predicted_depth.cpu().numpy()
        depth_map = (depth_map.max() - depth_map) + 0.1 # Invert and prevent division by zero
        
        # Normalize to a 0-255 scale for visualization, but keep raw for Open3D
        depth_map_normalized = cv2.normalize(depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        return np.array(image), depth_map, depth_map_normalized

class PointCloudProcessor:
    def __init__(self, K):
        self.K = K

    def depth_to_pointcloud(self, rgb_image, depth_map):
        """
        Converts the 2D depth map and RGB image into an Open3D Point Cloud.
        """
        # Open3D expects depth in uint16. We scale the relative depth map arbitrarily 
        # (e.g., by 1000) because the scale doesn't matter for finding plane intersections.
        depth_scaled = (depth_map * 1000).astype(np.uint16)
        
        color = o3d.geometry.Image(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        depth = o3d.geometry.Image(depth_scaled)
        
        # Depth scaling factor is set to 1000 to match our multiplication above
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_scale=1000.0, depth_trunc=1000.0, convert_rgb_to_intensity=False
        )
        
        h, w = rgb_image.shape[:2]
        intrinsics = o3d.camera.PinholeCameraIntrinsic(w, h, self.K[0,0], self.K[1,1], self.K[0,2], self.K[1,2])
        
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsics)
        
        # Downsample for faster RANSAC calculation
        pcd = pcd.voxel_down_sample(voxel_size=0.02)
        return pcd

class PlaneFitter:
    def extract_floor_and_walls(self, pcd, distance_threshold=0.05):
        """
        Uses RANSAC to iteratively find the 3 largest flat planes in the 3D point cloud.
        """
        # 1. Fit the largest plane (Assumption: This is the Floor)
        floor_eq, inliers_f = pcd.segment_plane(distance_threshold=distance_threshold,
                                                ransac_n=3,
                                                num_iterations=1000)
        remaining_pcd = pcd.select_by_index(inliers_f, invert=True)
        
        # 2. Fit the second largest plane (Assumption: Wall 1)
        wall1_eq, inliers_w1 = remaining_pcd.segment_plane(distance_threshold=distance_threshold,
                                                           ransac_n=3,
                                                           num_iterations=1000)
        remaining_pcd = remaining_pcd.select_by_index(inliers_w1, invert=True)

        # 3. Fit the third largest plane (Assumption: Wall 2)
        wall2_eq, inliers_w2 = remaining_pcd.segment_plane(distance_threshold=distance_threshold,
                                                           ransac_n=3,
                                                           num_iterations=1000)
        
        return floor_eq, wall1_eq, wall2_eq

class IntersectionSolver:
    def __init__(self, K):
        self.K = K

    def find_corner_3d(self, plane1_eq, plane2_eq, plane3_eq):
        """
        Solves the linear system to find the (X,Y,Z) intersection of 3 planes.
        Plane equation: Ax + By + Cz + D = 0 -> Ax + By + Cz = -D
        """
        # A matrix holds the normal vectors (A, B, C)
        A = np.array([plane1_eq[:3], plane2_eq[:3], plane3_eq[:3]])
        # B vector holds the -D values
        B = np.array([-plane1_eq[3], -plane2_eq[3], -plane3_eq[3]])
        
        try:
            corner_3d = np.linalg.solve(A, B)
            return corner_3d
        except np.linalg.LinAlgError:
            # Singular matrix means planes don't intersect at a single point (e.g. parallel)
            return None

    def project_to_2d(self, point_3d):
        """
        Projects a 3D point back into 2D pixel space using the intrinsic matrix.
        """
        point_3d_homog = np.array([[point_3d[0]], [point_3d[1]], [point_3d[2]]])
        point_2d_homog = self.K @ point_3d_homog
        
        u = int(point_2d_homog[0] / point_2d_homog[2])
        v = int(point_2d_homog[1] / point_2d_homog[2])
        return (u, v)

# --- Main Orchestration ---
if __name__ == "__main__":
    
    # ---------------- CONFIGURATION ----------------
    INPUT_DIR = "./Point_Detection_Tests"      
    OUTPUT_DIR = "./Point_Detection_Tests/annotated_depth"  
    
    # Example Camera Intrinsic Matrix (K)
    # You must replace this with the actual focal length and optical center of your camera!
    K_MATRIX = np.array([
        [800.0, 0.0,   640.0],
        [0.0,   800.0, 360.0],
        [0.0,   0.0,   1.0]
    ])
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    # -----------------------------------------------
    
    # 1. Initialize modules ONCE outside the loop
    depth_module = DepthAnythingModule()
    pcd_module = PointCloudProcessor(K_MATRIX)
    fitter = PlaneFitter()
    solver = IntersectionSolver(K_MATRIX)
    
    search_pattern = os.path.join(INPUT_DIR, "*.jpg")
    image_paths = glob.glob(search_pattern)
    
    if len(image_paths) == 0:
        print(f"No .jpg files found in '{INPUT_DIR}'. Please check the path.")
    else:
        print(f"Found {len(image_paths)} images. Starting batch processing...")

    # 2. Process each image
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")
        
        try:
            # Infer Depth
            rgb_img, raw_depth, vis_depth = depth_module.get_depth_map(img_path)
            
            # Convert to 3D Point Cloud
            pcd = pcd_module.depth_to_pointcloud(rgb_img, raw_depth)
            
            # Fit 3D Planes
            floor_eq, wall1_eq, wall2_eq = fitter.extract_floor_and_walls(pcd)
            
            # Mathematically Intersect Planes
            corner_3d = solver.find_corner_3d(floor_eq, wall1_eq, wall2_eq)
            
            if corner_3d is not None:
                corner_2d = solver.project_to_2d(corner_3d)
                print(f"  -> Projected 2D Pixel Coordinate: X:{corner_2d[0]}, Y:{corner_2d[1]}")
                
                # Visualize the result by drawing a red dot on the corner
                cv2.circle(rgb_img, corner_2d, radius=10, color=(0, 0, 255), thickness=-1)
                
                # Save both the annotated RGB image and the Depth map
                save_img_path = os.path.join(OUTPUT_DIR, f"annotated_{filename}")
                save_depth_path = os.path.join(OUTPUT_DIR, f"depth_{filename}")
                
                cv2.imwrite(save_img_path, cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(save_depth_path, vis_depth)
                
            else:
                print(f"  -> Warning: Planes are parallel and do not intersect for {filename}.")
                
        except Exception as e:
            print(f"  -> Error processing {filename}: {e}")
            
    print(f"\nBatch processing complete! Annotated images saved to '{OUTPUT_DIR}'")