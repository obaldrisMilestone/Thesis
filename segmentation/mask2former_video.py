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

        # ADE20K class indices
        self.WALL_CLASS_INDEX = 0
        self.FLOOR_CLASS_INDEX = 3 
        self.DOOR_CLASS_INDEX = 14 # <-- ADD THIS

    def process_video_masks(self, video_path, sample_interval_sec=1.0):
        """
        Reads a video, samples frames at the given interval, segments them, 
        and computes the temporal union of the masks.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate how many frames to skip to achieve the desired second interval
        frame_step = int(max(1, fps * sample_interval_sec))
        
        accumulated_floor = None
        accumulated_wall = None
        reference_image = None
        
        current_frame = 0
        samples_taken = 0

        while True:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            
            if not ret:
                break # End of video
                
            # Convert OpenCV BGR to PIL RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image_rgb)
            
            # Save the very first sampled frame as a visual reference
            if reference_image is None:
                reference_image = frame.copy()

            # --- 1. Run Mask2Former ---
            inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
                
            predicted_semantic_map = self.processor.post_process_semantic_segmentation(
                outputs, target_sizes=[pil_image.size[::-1]]
            )[0]
            
            # ... (Inside the while loop after running the model) ...
            predicted_classes = predicted_semantic_map.cpu().numpy()
            
            # Extract current binary masks
            current_floor = np.where(predicted_classes == self.FLOOR_CLASS_INDEX, 255, 0).astype(np.uint8)
            current_wall = np.where(predicted_classes == self.WALL_CLASS_INDEX, 255, 0).astype(np.uint8)
            current_door = np.where(predicted_classes == self.DOOR_CLASS_INDEX, 255, 0).astype(np.uint8) # <-- ADD THIS

            # --- 2. Temporal Aggregation (Logical OR) ---
            if accumulated_floor is None:
                accumulated_floor = current_floor
                accumulated_wall = current_wall
                accumulated_door = current_door # <-- ADD THIS
            else:
                accumulated_floor = cv2.bitwise_or(accumulated_floor, current_floor)
                accumulated_wall = cv2.bitwise_or(accumulated_wall, current_wall)
                accumulated_door = cv2.bitwise_or(accumulated_door, current_door) # <-- ADD THIS
                
            # ... (End of while loop) ...
            
        cap.release()
        
        # --- Clean up overlaps (Hierarchical Prioritization) ---
        # 1. Floor beats Wall (a wall doesn't lay flat)
        accumulated_wall[accumulated_floor == 255] = 0
        # 2. Door beats Wall (a door is an opening INSIDE a wall)
        accumulated_wall[accumulated_door == 255] = 0
        # 3. Floor beats Door (doors touch the floor, but the walkable area is floor)
        accumulated_door[accumulated_floor == 255] = 0 

        return reference_image, accumulated_floor, accumulated_wall, accumulated_door

# Don't forget to update the function signature to accept the door_mask!
def create_temporal_annotation(image, floor_mask, wall_mask, door_mask):
    colored_mask = np.zeros_like(image)
    
    # Floor is Green, Walls are Red, Doors are Cyan (Blue + Green)
    colored_mask[floor_mask == 255] = [0, 255, 0]
    colored_mask[wall_mask == 255] = [0, 0, 255]      # OpenCV uses BGR: (Blue, Green, Red)
    colored_mask[door_mask == 255] = [255, 255, 0]    # Cyan in BGR
    
    # Blend with the reference image
    overlay = cv2.addWeighted(image, 0.4, colored_mask, 0.6, 0)
    return overlay
# --- Main Execution ---
if __name__ == "__main__":
    
    # Point this to the folder containing your MP4 files 
    # (e.g., your Hospital_000_Val video folder)
    INPUT_DIR = "/home/user/thesis/code/Hospital_000_Val/MTMC_Tracking_2025/val/Hospital_000/videos"      
    OUTPUT_DIR = "./dataset/temporal_masks"  
    
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
            # We sample a frame every 2.0 seconds. 
            # You can increase this to 1.0 or 0.5 depending on how fast people move in the video.
            ref_img, final_floor, final_wall = segmenter.process_video_masks(vid_path, sample_interval_sec=10.0)
            
            if ref_img is not None:
                # Save the accumulated results
                annotated_bgr = create_temporal_annotation(ref_img, final_floor, final_wall)
                
                # Save as an image with the same name as the video
                save_name = filename.replace(".mp4", "_temporal_bg.jpg")
                save_path = os.path.join(OUTPUT_DIR, save_name)
                cv2.imwrite(save_path, annotated_bgr)
                print(f"  -> Saved temporal accumulation to {save_path}")
            else:
                print(f"  -> Could not read any frames from {filename}")
                
        except Exception as e:
            print(f"  -> Error processing {filename}: {e}")
            
    print(f"\nBatch processing complete! Temporal backgrounds saved to '{OUTPUT_DIR}'")