import h5py
import numpy as np
import matplotlib.pyplot as plt  # Added for saving the image

local_path = "/home/user/thesis/code/dataset/depth_map.h5"

background_depth = None

print("Extracting background depth map...")

with h5py.File(local_path, 'r') as f:
    keys = list(f.keys())
    total_frames = len(keys)
    
    for i, key in enumerate(keys):
        current_frame = f[key][:]
        
        if background_depth is None:
            background_depth = np.copy(current_frame)
            continue
            
        np.maximum(background_depth, current_frame, out=background_depth)
        
        if (i + 1) % 500 == 0:
            print(f"Processed {i + 1}/{total_frames} frames...")

print("Done!")
print(f"Final shape: {background_depth.shape}")

# 1. Save the raw array for future calculations or masking
npy_output_path = "/home/user/thesis/code/dataset/background_depth.npy"
np.save(npy_output_path, background_depth)
print(f"Saved raw data to {npy_output_path}")

# 2. Save the visualization image
image_output_path = "/home/user/thesis/code/dataset/background_depth_viz.png"
print("Generating visualization image...")

# plt.imsave automatically scales the depth values and applies the colormap
plt.imsave(image_output_path, background_depth, cmap='plasma')

print(f"Saved visualization to {image_output_path}")