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

    def process_video_depth(self, video_path, diff_threshold=0.15, alpha=0.1):
        """
        Extracts temporal depth by comparing only the first and last frames, 
        using proportional thresholding to ignore noise while catching structural reveals.
        
        Args:
            diff_threshold (float): The proportional difference required to consider 
                                    a pixel a "structural reveal" (e.g., 0.15 = 15%).
            alpha (float): The smoothing factor for static background noise.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine the exact frame indices for the first and last frame
        target_frame_indices = [0, max(0, total_frames - 1)]
        
        accumulated_disparity = None
        reference_image = None
        
        for frame_idx in target_frame_indices:
            # Seek to the specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Warning: Could not read frame {frame_idx} from {video_path}")
                continue 
                
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
                reveal_mask = proportional_diff > diff_threshold
                
                # Condition B: Tiny Fluctuation (Static background noise)
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

        cap.release()
        
        if accumulated_disparity is None:
            return None, None, None
            
        # --- 3. Post-Process for Open3D Compatibility ---
        # 1. Normalize the disparity strictly between 0 and 1 so the math is stable
        min_disp = accumulated_disparity.min()
        max_disp = accumulated_disparity.max()
        disp_normalized = (accumulated_disparity - min_disp) / (max_disp - min_disp + 1e-8)

        # 2. Invert to get true Relative Depth (Z = 1 / Disparity)
        # We add 0.05 to the denominator to prevent division-by-zero for pixels at infinity
        depth_map = 1.0 / (disp_normalized + 0.05)
        
        depth_map_normalized = cv2.normalize(
            depth_map, None, 0, 255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        depth_colored = cv2.applyColorMap(depth_map_normalized, cv2.COLORMAP_INFERNO)

        return reference_image, depth_map, depth_colored


# --- Main Execution ---
if __name__ == "__main__":
    
    INPUT_DIR = "/home/user/thesis/code/dataset/Hospital_000_Val/MTMC_Tracking_2025/val/Hospital_000/videos"      
    OUTPUT_DIR = "/home/user/thesis/code/depth/temporal_depth2"  
    
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
            # Process only the first and last frame
            ref_img, raw_depth, vis_depth = segmenter.process_video_depth(vid_path)
            
            if ref_img is not None:
                save_name_vis = filename.replace(".mp4", "_temporal_depth_vis.jpg")
                save_path_vis = os.path.join(OUTPUT_DIR, save_name_vis)
                cv2.imwrite(save_path_vis, vis_depth)
                
                # Save the raw float32 depth map as a .npy file
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