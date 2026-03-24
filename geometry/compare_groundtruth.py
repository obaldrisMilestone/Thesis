import json
import numpy as np

def evaluate_extrinsics(predicted_json_path, gt_json_path):
    # 1. Load Your Predictions
    with open(predicted_json_path, 'r') as f:
        predicted_data = json.load(f)
        
    # For this example, we assume you extracted the specific Camera_24 gt data
    with open(gt_json_path, 'r') as f:
        gt_data = json.load(f)

    # 2. Extract Ground Truth Data
    ext_matrix = np.array(gt_data["calibration"]["extrinsicMatrix"])
    R = ext_matrix[:, :3]
    t = ext_matrix[:, 3]

    # Ground Truth Normal (3rd column of R)
    gt_normal = R[:, 2]
    # Ensure it points towards the camera (negative Y) like your predictions do
    if gt_normal[1] > 0:
        gt_normal = -gt_normal
    
    # Ground Truth Height (Z component of -R^T * t)
    gt_camera_center = -np.dot(R.T, t)
    gt_height = abs(gt_camera_center[2])

    print(f"--- GROUND TRUTH ---")
    print(f"Normal: [{gt_normal[0]:.4f}, {gt_normal[1]:.4f}, {gt_normal[2]:.4f}]")
    print(f"Height: {gt_height:.3f} meters")

    # 3. Compare with Prediction (Assuming "Camera_24" is in your results)
    camera_id = gt_data["calibration"]["id"]
    print(camera_id)
    if camera_id in predicted_data:
        pred = predicted_data[camera_id]
        pred_normal = np.array(pred["floor_normal"])
        pred_height = pred["camera_height"]
        
        print(f"\n--- PREDICTIONS ---")
        print(f"Normal: [{pred_normal[0]:.4f}, {pred_normal[1]:.4f}, {pred_normal[2]:.4f}]")
        print(f"Height: {pred_height:.3f} units")

        # 4. Calculate Angular Error (Dot Product)
        # dot(a,b) = |a||b|cos(theta) -> theta = arccos(dot(a,b))
        dot_prod = np.dot(gt_normal, pred_normal)
        # Clip to [-1, 1] to avoid floating point errors with arccos
        dot_prod = np.clip(dot_prod, -1.0, 1.0)
        angle_error_rad = np.arccos(dot_prod)
        angle_error_deg = np.degrees(angle_error_rad)

        # 5. Calculate Height Error
        # Note: If your depth map wasn't metrically scaled, this error will be large.
        height_error = abs(gt_height - pred_height)

        print(f"\n--- ERRORS ---")
        print(f"Normal Angular Error: {angle_error_deg:.2f} degrees")
        print(f"Height Absolute Error: {height_error:.3f} meters")
        
        # Determine if the depth scale needs calibration
        if height_error > 0.5:
             scale_factor = gt_height / pred_height
             print(f"\n[Tip] Your depth network output is likely unscaled relative depth.")
             print(f"Multiply your depth maps by {scale_factor:.3f} to convert them to true meters.")

    else:
        print(f"\n[!] Camera {camera_id} not found in your predicted results.")

# Usage:
evaluate_extrinsics("/home/user/thesis/code/geometry/camera_extrinsics.json", "/home/user/thesis/code/dataset/Point_Detection_Tests/Camera_07_metadata.json")