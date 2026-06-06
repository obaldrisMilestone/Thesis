import os
import glob
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

class TemporalVideoSegmenter:
    def __init__(self, model_name="facebook/mask2former-swin-tiny-ade-semantic"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Mask2Former to {self.device}...")
        
        self.processor = Mask2FormerImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # ADE20K Structural Class Indices
        self.WALL_CLASS_INDEX = 0
        self.FLOOR_CLASS_INDEX = 3 
        self.CEILING_CLASS_INDEX = 5
        self.WINDOW_CLASS_INDEX = 8
        self.DOOR_CLASS_INDEX = 14
        self.STAIRS_CLASS_INDEX = 53

    def process_video_masks(self, video_path):
        """
        Reads a video, extracts ONLY the first and last frame, segments them, 
        and computes the temporal union of all structural masks.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Determine the exact frame indices for the first and last frame
        target_frame_indices = [0, max(0, total_frames - 1)]
        
        accumulated = {
            "floor": None,
            "wall": None,
            "ceiling": None,
            "window": None,
            "door": None,
            "stairs": None
        }
        
        reference_image = None

        for frame_idx in target_frame_indices:
            # Seek to the specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Warning: Could not read frame {frame_idx} from {video_path}")
                continue
                
            # Convert OpenCV BGR to PIL RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Save the first frame as the visual reference for blending
            if reference_image is None:
                reference_image = frame.copy()

            # --- 1. Run Mask2Former ---
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            predicted_semantic_map = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[pil_image.size[::-1]]
            )[0]
            
            predicted_classes = predicted_semantic_map.cpu().numpy()
            
            # Extract current binary masks for all structural elements
            current = {
                "floor": np.where(predicted_classes == self.FLOOR_CLASS_INDEX, 255, 0).astype(np.uint8),
                "wall": np.where(predicted_classes == self.WALL_CLASS_INDEX, 255, 0).astype(np.uint8),
                "ceiling": np.where(predicted_classes == self.CEILING_CLASS_INDEX, 255, 0).astype(np.uint8),
                "window": np.where(predicted_classes == self.WINDOW_CLASS_INDEX, 255, 0).astype(np.uint8),
                "door": np.where(predicted_classes == self.DOOR_CLASS_INDEX, 255, 0).astype(np.uint8),
                "stairs": np.where(predicted_classes == self.STAIRS_CLASS_INDEX, 255, 0).astype(np.uint8),
            }

            # --- 2. Temporal Aggregation (Logical OR) ---
            if accumulated["floor"] is None:
                for key in accumulated:
                    accumulated[key] = current[key]
            else:
                for key in accumulated:
                    accumulated[key] = cv2.bitwise_or(accumulated[key], current[key])
                
        cap.release()
        
        # Safety check if no frames were read
        if accumulated["floor"] is None:
            return None, None
        
        # --- 3. Clean up overlaps (Hierarchical Prioritization) ---
        # Floors and ceilings beat walls
        accumulated["wall"][accumulated["floor"] == 255] = 0
        accumulated["wall"][accumulated["ceiling"] == 255] = 0
        
        # Openings (doors and windows) beat walls because they exist inside them
        accumulated["wall"][accumulated["door"] == 255] = 0
        accumulated["wall"][accumulated["window"] == 255] = 0
        
        # Stairs beat both floors and walls
        accumulated["floor"][accumulated["stairs"] == 255] = 0
        accumulated["wall"][accumulated["stairs"] == 255] = 0

        return reference_image, accumulated

def create_temporal_annotation(image, masks):
    """
    Blends the hierarchical masks with the reference image using distinct colors.
    """
    colored_mask = np.zeros_like(image)
    
    # Define colors in BGR format for OpenCV
    colors = {
        "floor": [0, 255, 0],       # Green
        "wall": [0, 0, 255],        # Red
        "door": [255, 255, 0],      # Cyan
        "ceiling": [255, 0, 255],   # Magenta
        "window": [255, 0, 0],      # Blue
        "stairs": [0, 165, 255]     # Orange
    }
    
    # Apply colors based on the masks
    for key, mask in masks.items():
        colored_mask[mask == 255] = colors[key]
    
    # Blend with the reference image
    overlay = cv2.addWeighted(image, 0.4, colored_mask, 0.6, 0)
    return overlay

# --- Main Execution ---
if __name__ == "__main__":
    
    INPUT_DIR = "/home/user/thesis/code/dataset/Hospital_000_Val/MTMC_Tracking_2025/val/Hospital_000/videos"      
    OUTPUT_DIR = "./dataset/temporal_masks2"  
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    segmenter = TemporalVideoSegmenter()
    
    video_paths = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))
    
    if not video_paths:
        print(f"No .mp4 files found in '{INPUT_DIR}'.")
    else:
        print(f"Found {len(video_paths)} videos. Starting temporal processing...")

    for vid_path in video_paths:
        filename = os.path.basename(vid_path)
        print(f"\nProcessing Video: {filename}")
        
        try:
            # Process only the first and last frame
            ref_img, final_masks = segmenter.process_video_masks(vid_path)
            
            if ref_img is not None:
                # Save the accumulated results
                annotated_bgr = create_temporal_annotation(ref_img, final_masks)
                
                # Save as an image with the same name as the video
                save_name = filename.replace(".mp4", "_temporal_bg.jpg")
                save_path = os.path.join(OUTPUT_DIR, save_name)
                cv2.imwrite(save_path, annotated_bgr)
                # Save the raw binary masks for later depth filtering
                np.savez(save_path.replace(".jpg", ".npz"), **final_masks)
                print(f"  -> Saved temporal accumulation to {save_path}")
            else:
                print(f"  -> Could not read any frames from {filename}")
                
        except Exception as e:
            print(f"  -> Error processing {filename}: {e}")
            
    print(f"\nBatch processing complete! Temporal backgrounds saved to '{OUTPUT_DIR}'")