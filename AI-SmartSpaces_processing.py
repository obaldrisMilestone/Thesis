import cv2
import json
import os
from pathlib import Path

def prepare_point_detection_dataset(data_dir, output_dir):
    """
    Extracts one frame per video and creates a per-image calibration JSON.
    """
    video_dir = Path(data_dir) / "videos"
    calib_file = Path(data_dir) / "calibration.json"
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Load the main calibration file
    with open(calib_file, 'r') as f:
        main_calib = json.load(f)

    # Create a lookup for camera sensors by ID
    sensor_lookup = {s['id']: s for s in main_calib['sensors'] if s['type'] == 'camera'}

    # Process each video file
    video_files = list(video_dir.glob("*.mp4"))
    print(f"Found {len(video_files)} videos in {video_dir}")

    for video_path in video_files:
        # Expected filename format: 'Camera_01.mp4' or similar
        camera_id = video_path.stem 
        
        if camera_id not in sensor_lookup:
            print(f"Warning: No calibration found for {camera_id}. Skipping.")
            continue

        # 1. Extract the first frame
        cap = cv2.VideoCapture(str(video_path))
        success, frame = cap.read()
        cap.release()

        if not success:
            print(f"Error: Could not read frame from {video_path}")
            continue

        # 2. Save the image
        img_filename = f"{camera_id}_frame.jpg"
        img_output_path = out_path / img_filename
        cv2.imwrite(str(img_output_path), frame)

        # 3. Create individual calibration JSON for this image
        sensor_data = sensor_lookup[camera_id]
        image_metadata = {
            "source_video": video_path.name,
            "extracted_frame": img_filename,
            "calibration": sensor_data
        }

        json_output_path = out_path / f"{camera_id}_metadata.json"
        with open(json_output_path, 'w') as jf:
            json.dump(image_metadata, jf, indent=4)

        print(f"Processed {camera_id}: Saved image and metadata.")

# Example Usage
# data_dir should point to your downloaded Hospital_000_Val folder
prepare_point_detection_dataset(
    data_dir="./Hospital_000_Val/MTMC_Tracking_2025/val/Hospital_000",
    output_dir="./Point_Detection_Tests"
)



