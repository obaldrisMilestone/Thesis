import torch
import cv2
import numpy as np
from PIL import Image
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation

class Mask2FormerSegmentationModule:
    def __init__(self, model_name="facebook/mask2former-swin-tiny-ade-semantic"):
        """
        Loads a pre-trained Mask2Former model. 
        We use the 'swin-tiny' version for faster inference, but you can 
        upgrade to 'swin-large' for higher accuracy in your final thesis benchmark.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading Mask2Former to {self.device}...")
        
        self.processor = Mask2FormerImageProcessor.from_pretrained(model_name)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # ADE20K class index for 'floor, flooring'
        self.FLOOR_CLASS_INDEX = 3 

    def get_floor_mask(self, image_path):
        image = Image.open(image_path).convert("RGB")
        
        # Prepare inputs
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        # Mask2Former's processor handles the upsampling back to the original image size natively!
        # target_sizes expects a list of (height, width) tuples
        target_sizes = [image.size[::-1]]
        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=target_sizes
        )[0]
        
        # Convert to numpy and isolate the floor class
        predicted_classes = predicted_semantic_map.cpu().numpy()
        floor_mask = np.where(predicted_classes == self.FLOOR_CLASS_INDEX, 255, 0).astype(np.uint8)
        
        return np.array(image), floor_mask

class BoundaryExtractor:
    def __init__(self, epsilon_factor=0.015):
        # epsilon_factor controls polygon rigidity. 
        self.epsilon_factor = epsilon_factor

    def extract_corners(self, binary_mask):
        # 1. Find Contours (RETR_EXTERNAL to ignore holes/rugs)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None, []
            
        # Assume the largest contour by area is the main floor
        largest_contour = max(contours, key=cv2.contourArea)
        
        # 2. Polygon Approximation
        perimeter = cv2.arcLength(largest_contour, True)
        epsilon = self.epsilon_factor * perimeter
        approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        # Extract the (x, y) coordinates of the vertices
        corners = [point[0].tolist() for point in approx_polygon]
        
        return largest_contour, corners

def visualize_results(image, floor_mask, contour, corners):
    # Create a semi-transparent green overlay for the floor
    colored_mask = np.zeros_like(image)
    colored_mask[floor_mask == 255] = [0, 255, 0]
    overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

    # Draw the simplified polygon boundary (Blue)
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, (255, 0, 0), 2)

    # Draw the exact corner points (Red Dots)
    for (x, y) in corners:
        cv2.circle(overlay, (x, y), radius=8, color=(0, 0, 255), thickness=-1)
        
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imshow("Mask2Former Floor Layout", overlay_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# --- Main Execution ---
if __name__ == "__main__":
    # Point this to a frame from your MMPTrack or AICity'25 dataset
    INPUT_IMAGE = "test_room.jpg" 
    
    seg_module = Mask2FormerSegmentationModule()
    boundary_extractor = BoundaryExtractor(epsilon_factor=0.02)
    
    original_img, floor_mask = seg_module.get_floor_mask(INPUT_IMAGE)
    contour, corner_points = boundary_extractor.extract_corners(floor_mask)
    
    print(f"Detected {len(corner_points)} geometric corners using Mask2Former:")
    for idx, pt in enumerate(corner_points):
        print(f"  Corner {idx+1}: (X: {pt[0]}, Y: {pt[1]})")
        
    visualize_results(original_img, floor_mask, contour, corner_points)