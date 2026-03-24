import os
import glob
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

class TemporalDepthSegmenter:
    def __init__(self, model_name="LiheYoung/depth-anything-small-hf"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading DepthAnything to {self.device}...")
        
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def process_video_depth(self, video_path, sample_interval_sec=1.0, diff_threshold=0.15, alpha=0.1):
        """
        Extracts temporal depth using proportional thresholding to ignore noise 
        while catching structural reveals.
        
        Args:
            diff_threshold (float): The proportional difference required to consider 
                                    a pixel a "structural reveal" (e.g., 0.15 = 15%).
            alpha (float): The smoothing factor for static background noise.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_step = int(max(1, fps * sample_interval_sec))
        
        accumulated_disparity = None
        reference_image = None
        
        current_frame = 0
        samples_taken = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            
            if not ret:
                break 
                
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            if reference_image is None:
                reference_image = frame.copy()

            # --- 1. Infer Depth (Disparity) ---
            inputs = self.image_processor(images=pil_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                predicted_depth = outputs.predicted_depth
                
            predicted_depth = torch.nn.functional.interpolate(
                predicted_depth.unsqueeze(1),
                size=pil_image.size[::-1],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            current_disparity = predicted_depth.cpu().numpy()

            # --- 2. Intelligent Temporal Aggregation ---
            if accumulated_disparity is None:
                accumulated_disparity = current_disparity
            else:
                # Calculate the proportional difference
                # Positive value = The new pixel is FARTHER away than the accumulated one
                # Add a tiny epsilon to prevent division by zero
                proportional_diff = (accumulated_disparity - current_disparity) / (accumulated_disparity + 1e-5)
                
                # Condition A: Massive Drop (Obstacle moved out of the way)
                # If disparity drops by more than the threshold (e.g., 15%), it's a structural reveal.
                reveal_mask = proportional_diff > diff_threshold
                
                # Condition B: Tiny Fluctuation (Static background noise)
                # If the difference is between -15% and +15%, it's just model jitter.
                noise_mask = np.abs(proportional_diff) <= diff_threshold
                
                # Condition C: Massive Spike (Obstacle walked in)
                # proportional_diff < -diff_threshold. We do nothing here to preserve the background.

                # Apply the updates
                # 1. Accept the new background entirely
                accumulated_disparity[reveal_mask] = current_disparity[reveal_mask]
                
                # 2. Apply Exponential Moving Average (EMA) to smooth out the noise
                accumulated_disparity[noise_mask] = (
                    (1.0 - alpha) * accumulated_disparity[noise_mask] + 
                    (alpha) * current_disparity[noise_mask]
                )

            samples_taken += 1
            
            current_frame += frame_step
            
        cap.release()
        
        # --- 3. Post-Process for Open3D Compatibility ---
        depth_map = (accumulated_disparity.max() - accumulated_disparity) + 0.1 
        
        depth_map_normalized = cv2.normalize(
            depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        depth_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_INFERNO)

        return reference_image, depth_map, depth_colored


# --- Main Execution ---
if __name__ == "__main__":
    
    INPUT_DIR = "/home/user/thesis/code/Hospital_000_Val/MTMC_Tracking_2025/val/Hospital_000/videos"      
    OUTPUT_DIR = "./dataset/temporal_depth"  
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    segmenter = TemporalDepthSegmenter()
    
    video_paths = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))
    
    if not video_paths:
        print(f"No .mp4 files found in '{INPUT_DIR}'.")
    else:
        print(f"Found {len(video_paths)} videos. Starting temporal depth processing...")

    for vid_path in video_paths:
        filename = os.path.basename(vid_path)
        print(f"\nProcessing Video: {filename}")
        
        try:
            # Extract a frame every 2.0 seconds
            ref_img, raw_depth, vis_depth = segmenter.process_video_depth(vid_path, sample_interval_sec=15.0)
            
            if ref_img is not None:
                save_name_vis = filename.replace(".mp4", "_temporal_depth_vis.jpg")
                save_path_vis = os.path.join(OUTPUT_DIR, save_name_vis)
                cv2.imwrite(save_path_vis, vis_depth)
                
                # We also save the raw float32 depth map as a .npy file so you can load it 
                # directly into Open3D later without losing precision!
                save_name_raw = filename.replace(".mp4", "_temporal_depth_raw.npy")
                save_path_raw = os.path.join(OUTPUT_DIR, save_name_raw)
                np.save(save_path_raw, raw_depth)
                
                print(f"  -> Saved clean visual depth to {save_path_vis}")
                print(f"  -> Saved raw point-cloud data to {save_path_raw}")
            else:
                print(f"  -> Could not read any frames from {filename}")
                
        except Exception as e:
            print(f"  -> Error processing {filename}: {e}")
            
    print(f"\nBatch processing complete!")