import os
import glob
import numpy as np
import cv2

def clean_layout_depth(depth_npy_path, masks_npz_path, output_dir):
    """
    Loads raw depth and structural masks, combining them to erase all non-layout objects.
    """
    filename = os.path.basename(depth_npy_path).replace("_temporal_depth_raw.npy", "")
    print(f"Cleaning depth for: {filename}")
    
    # 1. Load the raw data
    raw_depth = np.load(depth_npy_path)
    structural_masks = np.load(masks_npz_path)
    
    # 2. Initialize a blank boolean mask matching the depth map dimensions
    master_layout_mask = np.zeros(raw_depth.shape, dtype=bool)
    
    # 3. Combine all structural classes into one master mask (Logical OR)
    for element_name in structural_masks.files:
        mask_array = structural_masks[element_name]
        
        # Mask arrays are saved with 255 representing the structure
        master_layout_mask = np.logical_or(master_layout_mask, mask_array == 255)
            
    # 4. Apply the mask to the depth map
    filtered_depth = np.copy(raw_depth)
    
    # Invert the mask (~): Everywhere that is NOT structure becomes 0.0 (invalid)
    filtered_depth[~master_layout_mask] = 0.0
    
    # --- 5. Save the Raw Cleaned Data for 3D processing ---
    save_raw_path = os.path.join(output_dir, f"{filename}_clean_layout_depth.npy")
    np.save(save_raw_path, filtered_depth)
    
    # --- 6. Create and Save a Visual Debug Image ---
    valid_pixels = filtered_depth[filtered_depth > 0]
    
    if len(valid_pixels) > 0:
        min_disp = valid_pixels.min()
        max_disp = valid_pixels.max()
        
        # Normalize strictly ignoring the 0.0 invalid pixels
        depth_normalized = np.zeros_like(filtered_depth, dtype=np.uint8)
        depth_normalized[filtered_depth > 0] = (
            (filtered_depth[filtered_depth > 0] - min_disp) / (max_disp - min_disp + 1e-8) * 255
        )
        
        # Apply the inferno colormap
        vis_filtered_depth = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_INFERNO)
        
        # Turn the invalid non-structural pixels completely black
        vis_filtered_depth[filtered_depth == 0.0] = [0, 0, 0] 
        
        save_vis_path = os.path.join(output_dir, f"{filename}_clean_layout_depth_vis.jpg")
        cv2.imwrite(save_vis_path, vis_filtered_depth)
        
    print(f"  -> Saved clean layout to {output_dir}")

# --- Main Execution ---
if __name__ == "__main__":
    
    # Folders where your previous scripts saved their outputs
    DEPTH_DIR = "/home/user/thesis/code/depth/temporal_depth2"
    MASK_DIR = "/home/user/thesis/code/segmentation/temporal_masks2"
    OUTPUT_DIR = "/home/user/thesis/code/depth/temporal_depth2_cleaned"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Find all raw depth files
    depth_files = glob.glob(os.path.join(DEPTH_DIR, "*_temporal_depth_raw.npy"))
    
    if not depth_files:
        print(f"No raw depth .npy files found in {DEPTH_DIR}")
    else:
        print(f"Found {len(depth_files)} depth maps to clean.")
        
    for depth_path in depth_files:
        # Construct the expected mask filename based on the depth filename
        base_name = os.path.basename(depth_path).replace("_temporal_depth_raw.npy", "")
        mask_path = os.path.join(MASK_DIR, f"{base_name}_temporal_bg.npz")
        
        if os.path.exists(mask_path):
            try:
                clean_layout_depth(depth_path, mask_path, OUTPUT_DIR)
            except Exception as e:
                print(f"Error processing {base_name}: {e}")
        else:
            print(f"Warning: Could not find matching mask file for {base_name} at {mask_path}")
            
    print("\nBatch cleaning complete!")