import numpy as np
import open3d as o3d
import cv2

# --- 1. 3D Extraction Functions ---

def unproject_to_3d(depth_map, mask, intrinsic_matrix):
    """Converts 2D masked depth pixels into a 3D point cloud."""
    h, w = depth_map.shape
    fx, fy = intrinsic_matrix[0, 0], intrinsic_matrix[1, 1]
    cx, cy = intrinsic_matrix[0, 2], intrinsic_matrix[1, 2]
    
    v, u = np.where(mask == 255)
    z = depth_map[v, u]
    
    valid = z > 0
    u, v, z = u[valid], v[valid], z[valid]
    
    x = (u - cx) * z / fx
    y = (v - cy) * z / fy
    
    points = np.vstack((x, y, z)).T
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    return pcd

def find_plane_ransac(pcd, distance_threshold=0.05, num_iterations=1000):
    """Finds the dominant plane in a point cloud using RANSAC."""
    plane_model, inliers = pcd.segment_plane(
        distance_threshold=distance_threshold,
        ransac_n=3,
        num_iterations=num_iterations
    )
    return plane_model, inliers

def intersect_3_planes(plane1, plane2, plane3):
    """Calculates the 3D intersection point of 3 planes."""
    normals = np.array([plane1[:3], plane2[:3], plane3[:3]])
    constants = np.array([-plane1[3], -plane2[3], -plane3[3]])
    
    try:
        corner_point = np.linalg.solve(normals, constants)
        return corner_point
    except np.linalg.LinAlgError:
        return None

def extract_room_corners(depth_map, floor_mask, wall_mask, intrinsic_matrix):
    corners_3d = []

    # Adaptive thresholds — 5% of median scene depth.
    # Permissive enough for one RANSAC iteration to capture a full wall surface
    # (tight thresholds cause micro-patches, burning all iterations on one wall).
    # The duplicate-normal filter below prevents false corners from over-grouping.
    valid_z = depth_map[depth_map > 0]
    if len(valid_z) == 0:
        return []
    median_z = float(np.median(valid_z))
    distance_threshold = median_z * 0.05
    normal_radius      = median_z * 0.05

    # 1. Extract Floor
    floor_pcd = unproject_to_3d(depth_map, floor_mask, intrinsic_matrix)
    if len(floor_pcd.points) < 100:
        return [] # Not enough floor points
    floor_plane, _ = find_plane_ransac(floor_pcd, distance_threshold=distance_threshold)

    # 2. Extract Walls
    wall_pcd = unproject_to_3d(depth_map, wall_mask, intrinsic_matrix)
    wall_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=normal_radius, max_nn=30))
    
    wall_planes = []
    remaining_wall_pcd = wall_pcd

    max_walls = 4
    for i in range(max_walls):
        if len(remaining_wall_pcd.points) < 1000:
            break

        plane_model, inliers = find_plane_ransac(remaining_wall_pcd, distance_threshold=distance_threshold)
        remaining_wall_pcd = remaining_wall_pcd.select_by_index(inliers, invert=True)

        # Discard if a parallel plane in the same direction was already found
        # (handles parallel corridor walls producing duplicate normals)
        n_new = np.array(plane_model[:3])
        if any(np.dot(n_new, np.array(p[:3])) > 0.95 for p in wall_planes):
            continue
        wall_planes.append(plane_model)
        
    # 3. Find Intersections
    for i in range(len(wall_planes)):
        for j in range(i + 1, len(wall_planes)):
            wall_a = wall_planes[i]
            wall_b = wall_planes[j]
            
            corner_pt = intersect_3_planes(floor_plane, wall_a, wall_b)
            
            if corner_pt is not None:
                normal_a = np.array(wall_a[:3])
                normal_b = np.array(wall_b[:3])
                angle = np.abs(np.dot(normal_a, normal_b))
                
                if angle < 0.5: # Roughly perpendicular walls
                    # Ensure the corner is in front of the camera (Z > 0)
                    if corner_pt[2] > 0: 
                        corners_3d.append(corner_pt)

    return corners_3d


# --- 2. 2D Projection & Visualization Functions ---

def project_3d_to_2d(point_3d, intrinsic_matrix):
    """
    Projects a 3D point (X, Y, Z) back to 2D pixel coordinates (u, v)
    using the camera intrinsic matrix.
    """
    # 1. Multiply K * [X, Y, Z]^T
    point_projected = np.dot(intrinsic_matrix, point_3d)
    
    # 2. Divide by Z to convert from homogeneous to pixel coordinates
    z = point_projected[2]
    u = int(round(point_projected[0] / z))
    v = int(round(point_projected[1] / z))
    
    return (u, v)

def visualize_corners_on_image(image_bgr, corners_3d, intrinsic_matrix):
    """
    Takes the original image, projects the 3D corners onto it, and draws them.
    """
    output_img = image_bgr.copy()
    
    h, w = output_img.shape[:2]
    
    for corner in corners_3d:
        # Get the 2D pixel coordinate
        u, v = project_3d_to_2d(corner, intrinsic_matrix)
        
        # Check if the calculated pixel actually falls within the image bounds
        if 0 <= u < w and 0 <= v < h:
            # Draw a bright cyan circle with a dark border for high visibility
            cv2.circle(output_img, (u, v), radius=10, color=(255, 255, 0), thickness=-1) # Cyan fill
            cv2.circle(output_img, (u, v), radius=10, color=(0, 0, 0), thickness=2)      # Black border
            
            # Optional: Label the coordinates next to the point
            label = f"({u},{v})"
            cv2.putText(output_img, label, (u + 15, v - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(output_img, label, (u + 15, v - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

    return output_img


# --- Main Execution Example ---
if __name__ == "__main__":
    # 1. Mock Data Setup (Replace with your actual loaded files)
    
    # Intrinsic Matrix from your earlier JSON example
    K = np.array([
        [1662.7688, 0.0, 960.0],
        [0.0, 1662.7688, 540.0],
        [0.0, 0.0, 1.0]
    ])
    
    # Load your files
    depth_map = np.load("/home/user/thesis/code/depth/temporal_depth2/Camera_06_temporal_depth_raw.npy")  # Replace with your actual depth .npy file
    masks = np.load("/home/user/thesis/code/segmentation/temporal_masks2/Camera_06_temporal_bg.npz")
    floor_mask = masks["floor"]
    wall_mask = masks["wall"]
    original_image = cv2.imread("/home/user/thesis/code/dataset/Point_Detection_Tests/Camera_06_frame.jpg")
    
    # ... (Assuming variables are loaded) ...
    corners_3d = extract_room_corners(depth_map, floor_mask, wall_mask, K)
    
    print(f"Found {len(corners_3d)} valid corners.")
    
    # 2. Draw the corners on the image
    result_image = visualize_corners_on_image(original_image, corners_3d, K)
    
    # 3. Save the result
    cv2.imwrite("detected_corners_visual.jpg", result_image)
    print("Saved visual result to detected_corners_visual.jpg")