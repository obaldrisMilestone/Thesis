import os
import glob
import cv2
import numpy as np
import torch
import json
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

class WallInstanceExtractor:
    def __init__(self, sam_checkpoint, device="cuda"):
        """Initializes the Segment Anything Model."""
        print(f"Loading SAM model to {device}... (This takes a moment)")
        model_type = "vit_h"
        
        if not os.path.exists(sam_checkpoint):
            raise FileNotFoundError(f"SAM checkpoint not found at {sam_checkpoint}. Please download it.")
            
        self.sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        self.sam.to(device=device)
        
        # We tune SAM specifically to look for large structural surfaces 
        # and ignore tiny details like doorknobs or posters.
        self.mask_generator = SamAutomaticMaskGenerator(
            model=self.sam,
            points_per_side=32,
            pred_iou_thresh=0.86,
            stability_score_thresh=0.92,
            crop_n_layers=0, 
            min_mask_region_area=5000, # Ignore anything smaller than a large poster
        )

    def extract_semantic_wall_mask(self, img_path):
        """Re-uses our HSV logic to isolate the Red 'Wall' class."""
        img = cv2.imread(img_path)
        if img is None: return None
            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_red1, upper_red1 = np.array([0, 50, 50]), np.array([10, 255, 255])
        lower_red2, upper_red2 = np.array([170, 50, 50]), np.array([180, 255, 255])
        
        semantic_mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_red1, upper_red1),
            cv2.inRange(hsv, lower_red2, upper_red2)
        )
        return semantic_mask

    def process_image(self, rgb_path, semantic_path):
        """Runs SAM and filters the results using the semantic mask."""
        # Load RGB for SAM
        image_bgr = cv2.imread(rgb_path)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        
        # Load Semantic Mask
        semantic_mask = self.extract_semantic_wall_mask(semantic_path)
        if semantic_mask is None:
            return None, None
            
        # Match sizes if needed
        if image_rgb.shape[:2] != semantic_mask.shape[:2]:
            semantic_mask = cv2.resize(semantic_mask, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_NEAREST)

        # Generate SAM Masks
        sam_results = self.mask_generator.generate(image_rgb)
        
        wall_instances = []
        
        # Filter Logic: Intersection over SAM Area
        for result in sam_results:
            # SAM returns a boolean mask (True/False)
            sam_bool_mask = result['segmentation'] 
            
            # Convert to uint8 for OpenCV bitwise operations (0 or 255)
            sam_uint8_mask = (sam_bool_mask * 255).astype(np.uint8)
            
            # Find where SAM mask overlaps with our Semantic Wall Mask
            overlap = cv2.bitwise_and(sam_uint8_mask, semantic_mask)
            
            overlap_area = np.sum(overlap > 0)
            sam_area = np.sum(sam_bool_mask)
            
            # If 85% of this SAM region lands on a Red Wall pixel, we keep it!
            if sam_area > 0 and (overlap_area / sam_area) > 0.85:
                wall_instances.append(sam_uint8_mask)
                
        return wall_instances, image_bgr

    def visualize_and_save(self, image_bgr, wall_instances, output_vis_path, output_dir, camera_id):
        """Creates a color-coded visualization and saves individual masks."""
        output_vis = image_bgr.copy()
        
        # High-contrast colors for different walls
        colors = [
            (0, 255, 0),   # Green
            (255, 0, 0),   # Blue
            (0, 255, 255), # Yellow
            (255, 0, 255), # Magenta
            (0, 165, 255)  # Orange
        ]
        
        for i, w_mask in enumerate(wall_instances):
            # 1. Save individual binary mask for downstream CAD alignment
            instance_filename = os.path.join(output_dir, f"{camera_id}_wall_{i}.png")
            cv2.imwrite(instance_filename, w_mask)
            
            # 2. Add to visualization overlay
            color = colors[i % len(colors)]
            colored_mask = np.zeros_like(output_vis)
            colored_mask[w_mask == 255] = color
            
            # Blend it
            output_vis = cv2.addWeighted(output_vis, 1.0, colored_mask, 0.5, 0)
            
        cv2.imwrite(output_vis_path, output_vis)


# --- Main Batch Execution ---
if __name__ == "__main__":
    
    # Paths
    RGB_DIR = "/home/user/thesis/code/dataset/Point_Detection_Tests"
    SEMANTIC_DIR = "/home/user/thesis/code/segmentation/temporal_masks"
    OUTPUT_DIR = "/home/user/thesis/code/geometry/wall_instances"
    SAM_CHECKPOINT = "/home/user/thesis/code/sam_vit_h_4b8939.pth"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using compute device: {device.upper()}")
    
    try:
        extractor = WallInstanceExtractor(SAM_CHECKPOINT, device)
    except FileNotFoundError as e:
        print(e)
        exit(1)

    # Find all semantic masks
    semantic_files = glob.glob(os.path.join(SEMANTIC_DIR, "*_temporal_bg.jpg"))
    results_summary = {}

    print(f"\nStarting SAM Instance Extraction for {len(semantic_files)} cameras...\n")

    for sem_path in semantic_files:
        camera_id = os.path.basename(sem_path).replace("_temporal_bg.jpg", "")
        print(f"Processing {camera_id}...")
        
        # We need the original, clean RGB frame for SAM
        rgb_path = os.path.join(RGB_DIR, f"{camera_id}_frame.jpg")
        
        if not os.path.exists(rgb_path):
            print(f"  -> Missing RGB frame at {rgb_path}, skipping.")
            continue
            
        try:
            # Run the extraction
            wall_instances, image_bgr = extractor.process_image(rgb_path, sem_path)
            
            if wall_instances is not None:
                print(f"  -> Found {len(wall_instances)} distinct wall instances.")
                
                vis_path = os.path.join(OUTPUT_DIR, f"{camera_id}_instances_vis.jpg")
                extractor.visualize_and_save(image_bgr, wall_instances, vis_path, OUTPUT_DIR, camera_id)
                
                results_summary[camera_id] = len(wall_instances)
                
        except Exception as e:
            print(f"  -> Error: {e}")

    # Save a quick JSON summary of how many walls were found per camera
    summary_path = os.path.join(OUTPUT_DIR, "instance_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(results_summary, f, indent=4)
        
    print(f"\nBatch processing complete! Check the '{OUTPUT_DIR}' folder for your visuals and masks.")