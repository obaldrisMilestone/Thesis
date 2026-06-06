import os
import glob
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

class PanopticVideoSegmenter:
    def __init__(self, model_name="facebook/mask2former-swin-tiny-ade-semantic"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Mask2Former to {self.device}...")
        
        self.processor = Mask2FormerImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # ADE20K class index
        self.WALL_CLASS_INDEX = 0

    def get_iou(self, mask1, mask2):
        """Calculates the Intersection over Union (IoU) of two binary masks."""
        intersection = np.logical_and(mask1 == 255, mask2 == 255).sum()
        union = np.logical_or(mask1 == 255, mask2 == 255).sum()
        if union == 0:
            return 0.0
        return intersection / union

    def extract_frame_walls(self, frame_bgr):
        """Runs Panoptic Segmentation on a single frame and returns a list of wall masks."""
        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)

        inputs = self.processor(images=pil_image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        panoptic_results = self.processor.post_process_panoptic_segmentation(
            outputs, target_sizes=[pil_image.size[::-1]]
        )[0]
        
        panoptic_map = panoptic_results["segmentation"].cpu().numpy()
        segments_info = panoptic_results["segments_info"]
        
        wall_masks = []
        for segment in segments_info:
            if segment["label_id"] == self.WALL_CLASS_INDEX:
                instance_id = segment["id"]
                wall_mask = np.where(panoptic_map == instance_id, 255, 0).astype(np.uint8)
                wall_masks.append(wall_mask)
                
        return wall_masks

    def process_video_panoptic(self, video_path, iou_threshold=0.3):
        """
        Extracts walls from the first and last frame, tracking and merging 
        the instances across time using IoU matching.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame_indices = [0, max(0, total_frames - 1)]
        
        accumulated_walls = []
        reference_image = None

        for idx, frame_idx in enumerate(target_frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            
            if not ret:
                continue
                
            if reference_image is None:
                reference_image = frame.copy()

            # Extract distinct wall instances for this specific frame
            current_frame_walls = self.extract_frame_walls(frame)

            # --- Temporal Instance Matching ---
            if idx == 0:
                # First frame: just initialize the master list
                accumulated_walls = current_frame_walls
            else:
                # Last frame: match to existing walls or add as new
                for current_wall in current_frame_walls:
                    best_iou = 0.0
                    best_match_idx = -1
                    
                    for acc_idx, acc_wall in enumerate(accumulated_walls):
                        iou = self.get_iou(current_wall, acc_wall)
                        if iou > best_iou:
                            best_iou = iou
                            best_match_idx = acc_idx
                            
                    # If it overlaps significantly with an existing wall, merge them (erases moving people)
                    if best_iou > iou_threshold:
                        accumulated_walls[best_match_idx] = cv2.bitwise_or(
                            accumulated_walls[best_match_idx], current_wall
                        )
                    else:
                        # It's a completely newly revealed wall instance
                        accumulated_walls.append(current_wall)
                        
        cap.release()
        
        # Convert list of masks into a named dictionary for easy saving
        final_wall_dict = {f"wall_instance_{i}": mask for i, mask in enumerate(accumulated_walls)}
        
        return reference_image, final_wall_dict


def visualize_panoptic_walls(image, wall_dict):
    """Overlays each separated wall with a distinct, random color."""
    colored_overlay = np.zeros_like(image)
    
    # We seed the random generator based on the number of walls so 
    # the colors are somewhat consistent if you run it multiple times
    np.random.seed(42) 
    
    for wall_name, mask in wall_dict.items():
        # Generate random BGR color
        color = np.random.randint(50, 255, size=3).tolist()
        colored_overlay[mask == 255] = color
        
    return cv2.addWeighted(image, 0.5, colored_overlay, 0.5, 0)


# --- Main Execution ---
if __name__ == "__main__":
    INPUT_DIR = "/home/user/thesis/code/dataset/Hospital_000_Val/MTMC_Tracking_2025/val/Hospital_000/videos"      
    OUTPUT_DIR = "/home/user/thesis/code/segmentation/panoptic_walls"  
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    segmenter = PanopticVideoSegmenter()
    
    video_paths = glob.glob(os.path.join(INPUT_DIR, "*.mp4"))
    
    if not video_paths:
        print(f"No .mp4 files found in '{INPUT_DIR}'.")
    else:
        print(f"Found {len(video_paths)} videos. Starting panoptic processing...")

    for vid_path in video_paths:
        filename = os.path.basename(vid_path)
        print(f"\nProcessing Video: {filename}")
        
        try:
            ref_img, final_walls_dict = segmenter.process_video_panoptic(vid_path)
            
            if ref_img is not None:
                # 1. Save the visual blend
                annotated_bgr = visualize_panoptic_walls(ref_img, final_walls_dict)
                save_name_vis = filename.replace(".mp4", "_panoptic_walls.jpg")
                cv2.imwrite(os.path.join(OUTPUT_DIR, save_name_vis), annotated_bgr)
                
                # 2. Save the raw NumPy data for the depth filtering script
                save_name_npz = filename.replace(".mp4", "_panoptic_walls.npz")
                np.savez(os.path.join(OUTPUT_DIR, save_name_npz), **final_walls_dict)
                
                print(f"  -> Extracted {len(final_walls_dict)} distinct wall instances.")
                print(f"  -> Saved visualization to {save_name_vis}")
            else:
                print(f"  -> Could not read any frames from {filename}")
                
        except Exception as e:
            print(f"  -> Error processing {filename}: {e}")
            
    print(f"\nBatch processing complete! Separated wall data saved to '{OUTPUT_DIR}'")