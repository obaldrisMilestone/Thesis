import os
import glob
import torch
import cv2
import numpy as np
from PIL import Image
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

class SemanticSegmentationModule:
    def __init__(self, model_name="nvidia/segformer-b2-finetuned-ade-512-512"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading SegFormer to {self.device}...")
        self.image_processor = SegformerImageProcessor.from_pretrained(model_name)
        self.model = SegformerForSemanticSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # ADE20K indices
        self.WALL_CLASS_INDEX = 0
        self.FLOOR_CLASS_INDEX = 3 

    def get_masks(self, image_path):
        image = Image.open(image_path).convert("RGB")
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        upsampled_logits = torch.nn.functional.interpolate(
            outputs.logits, size=image.size[::-1], mode="bilinear", align_corners=False
        )
        predicted_classes = upsampled_logits.argmax(dim=1).squeeze().cpu().numpy()
        
        # Extract strict binary masks
        floor_mask = np.where(predicted_classes == self.FLOOR_CLASS_INDEX, 255, 0).astype(np.uint8)
        wall_mask = np.where(predicted_classes == self.WALL_CLASS_INDEX, 255, 0).astype(np.uint8)
        
        return np.array(image), floor_mask, wall_mask


class StructuralRegionGrower:
    def __init__(self):
        pass

    def grow_regions(self, floor_mask, wall_mask):
        """
        Preserves the original masks perfectly. 
        Calculates distance transforms to assign unclassified pixels (people, chairs) 
        to either the floor or the wall based on which is closer.
        """
        # 1. Invert masks because distanceTransform calculates distance to 0-value pixels
        inv_floor = cv2.bitwise_not(floor_mask)
        inv_wall = cv2.bitwise_not(wall_mask)
        
        # 2. Calculate Euclidean distance from every pixel to the nearest floor/wall
        dist_to_floor = cv2.distanceTransform(inv_floor, cv2.DIST_L2, 5)
        dist_to_wall = cv2.distanceTransform(inv_wall, cv2.DIST_L2, 5)
        
        # 3. Identify pixels that belong to NEITHER floor nor wall (the obstacles)
        known_mask = cv2.bitwise_or(floor_mask, wall_mask)
        unknown_mask = cv2.bitwise_not(known_mask)
        
        # 4. Create copies to preserve the exact original segmentation
        final_floor = floor_mask.copy()
        final_wall = wall_mask.copy()
        
        # 5. Fill only the unknown pixels based on minimum distance
        # The parentheses around (unknown_mask == 255) are strictly required!
        final_floor[(unknown_mask == 255) & (dist_to_floor < dist_to_wall)] = 255
        final_wall[(unknown_mask == 255) & (dist_to_wall <= dist_to_floor)] = 255
        
        return final_floor, final_wall

def create_partitioned_annotation(image, final_floor, final_wall):
    colored_mask = np.zeros_like(image)
    
    # Floor is Green, Walls are Red
    colored_mask[final_floor == 255] = [0, 255, 0]
    colored_mask[final_wall == 255] = [255, 0, 0] 
    
    # Blend heavily so we can see the geometry beneath
    overlay = cv2.addWeighted(image, 0.4, colored_mask, 0.6, 0)

    return cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
# --- Main Execution ---
if __name__ == "__main__":
    INPUT_DIR = "./Point_Detection_Tests"      
    OUTPUT_DIR = "./Point_Detection_Tests/partitioned_outputs"  
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    seg_module = SemanticSegmentationModule()
    region_grower = StructuralRegionGrower()
    
    image_paths = glob.glob(os.path.join(INPUT_DIR, "*.jpg"))
    print(f"Found {len(image_paths)} images. Starting batch processing...")

    for img_path in image_paths:
        filename = os.path.basename(img_path)
        print(f"Processing {filename}...")
        
        try:
            # 1. Get initial imperfect masks
            original_img, floor_mask, wall_mask = seg_module.get_masks(img_path)
            
            # 2. Mathematically grow the regions to fill occlusions
            final_floor, final_wall = region_grower.grow_regions(floor_mask, wall_mask)
            
            # 3. Save the clean structural output
            annotated_bgr = create_partitioned_annotation(original_img, final_floor, final_wall)
            
            save_path = os.path.join(OUTPUT_DIR, f"partitioned_{filename}")
            cv2.imwrite(save_path, annotated_bgr)
            
        except Exception as e:
            print(f"  -> Error processing {filename}: {e}")