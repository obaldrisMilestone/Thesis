import json
import numpy as np
import glob
import os

def evaluate_batch_extrinsics(predicted_json_path, gt_dir_path, output_summary_path):
    # 1. Load All Predictions
    if not os.path.exists(predicted_json_path):
        print(f"Prediction file not found: {predicted_json_path}")
        return

    with open(predicted_json_path, 'r') as f:
        predicted_data = json.load(f)

    # 2. Find all Ground Truth metadata files in the directory
    search_pattern = os.path.join(gt_dir_path, "*_metadata.json")
    gt_files = glob.glob(search_pattern)
    
    if not gt_files:
        print(f"No ground truth metadata files found in {gt_dir_path}")
        return

    # Dictionary to store our results
    summary_results = {
        "aggregate": {},
        "per_camera": {}
    }

    angular_errors = []
    scale_factors = []

    print(f"Starting batch evaluation for {len(gt_files)} cameras...\n")

    # 3. Process each camera
    for gt_file in gt_files:
        with open(gt_file, 'r') as f:
            gt_data = json.load(f)

        # Safely extract camera ID
        camera_id = gt_data.get("calibration", {}).get("id")
        if not camera_id:
            # Fallback if the JSON structure is slightly different
            camera_id = os.path.basename(gt_file).replace("_metadata.json", "")

        if camera_id not in predicted_data:
            print(f"[!] Camera {camera_id} not found in predicted results. Skipping.")
            continue

        # Extract Ground Truth Data
        ext_matrix = np.array(gt_data["calibration"]["extrinsicMatrix"])
        R = ext_matrix[:, :3]
        t = ext_matrix[:, 3]

        # Ground Truth Normal (3rd column of R)
        gt_normal = R[:, 2]
        if gt_normal[1] > 0:
            gt_normal = -gt_normal
        
        # Ground Truth Height
        gt_camera_center = -np.dot(R.T, t)
        gt_height = abs(gt_camera_center[2])

        # Extract Prediction
        pred = predicted_data[camera_id]
        pred_normal = np.array(pred["floor_normal"])
        pred_height = pred["camera_height"]

        # Calculate Angular Error
        dot_prod = np.dot(gt_normal, pred_normal)
        dot_prod = np.clip(dot_prod, -1.0, 1.0) # Prevent arccos NaN errors
        angle_error_deg = float(np.degrees(np.arccos(dot_prod)))

        # Calculate Height Error & Scale Factor
        height_error = float(abs(gt_height - pred_height))
        scale_factor = float(gt_height / pred_height) if pred_height > 0 else 0.0

        # Store individual results (cast everything to float for JSON serialization)
        summary_results["per_camera"][camera_id] = {
            "gt_normal": [float(x) for x in gt_normal],
            "pred_normal": [float(x) for x in pred_normal],
            "angular_error_deg": angle_error_deg,
            "gt_height_m": float(gt_height),
            "pred_height_units": float(pred_height),
            "suggested_scale_factor": scale_factor
        }

        angular_errors.append(angle_error_deg)
        scale_factors.append(scale_factor)

        print(f"Processed {camera_id}: Error = {angle_error_deg:05.2f}°, Scale = {scale_factor:.3f}")

    # 4. Calculate Aggregate Metrics
    if angular_errors:
        summary_results["aggregate"] = {
            "total_evaluated": len(angular_errors),
            "mean_angular_error_deg": float(np.mean(angular_errors)),
            "median_angular_error_deg": float(np.median(angular_errors)),
            "mean_suggested_scale_factor": float(np.mean(scale_factors)),
            "median_suggested_scale_factor": float(np.median(scale_factors))
        }

    # 5. Write to Summary JSON
    with open(output_summary_path, 'w') as f:
        json.dump(summary_results, f, indent=4)

    print(f"\n--- BATCH SUMMARY ---")
    if angular_errors:
        print(f"Evaluated {len(angular_errors)} cameras.")
        print(f"Mean Angular Error: {summary_results['aggregate']['mean_angular_error_deg']:.2f} degrees")
        print(f"Mean Depth Scale Factor: {summary_results['aggregate']['mean_suggested_scale_factor']:.3f}x")
    print(f"Full details saved to: {output_summary_path}")


if __name__ == "__main__":
    # Define your paths here
    PREDICTED_EXTRINSICS = "/home/user/thesis/code/geometry/camera_extrinsics.json"
    GT_DIRECTORY = "/home/user/thesis/code/dataset/Point_Detection_Tests"
    OUTPUT_SUMMARY = "/home/user/thesis/code/geometry/evaluation_summary.json"

    evaluate_batch_extrinsics(PREDICTED_EXTRINSICS, GT_DIRECTORY, OUTPUT_SUMMARY)