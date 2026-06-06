import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
from huggingface_hub import HfFileSystem
from concurrent.futures import ThreadPoolExecutor, as_completed

# Enable Hugging Face's Rust-based fast transfer library
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

# --- Configuration ---
REPO_ID = "datasets/nvidia/PhysicalAI-SmartSpaces"
REMOTE_BASE_DIR = f"{REPO_ID}/MTMC_Tracking_2025"
LOCAL_BASE_DIR = "/home/user/thesis/code/dataset" 

# How many files to process at exactly the same time (adjust based on your CPU/RAM/Internet)
MAX_WORKERS = 4 

def extract_and_cleanup_depth(h5_local_path):
    """Processes the .h5 depth map in chunks, saves outputs, and deletes the .h5"""
    background_depth = None
    chunk_size = 50 # Load 50 frames at a time
    
    with h5py.File(h5_local_path, 'r') as f:
        keys = list(f.keys())
        
        # Process in chunks to reduce Python loop overhead
        for i in range(0, len(keys), chunk_size):
            chunk_keys = keys[i:i + chunk_size]
            
            # Load the chunk of frames into a single 3D numpy array
            frames_chunk = np.array([f[k][:] for k in chunk_keys])
            
            # Find the max depth across this specific chunk
            chunk_max = np.max(frames_chunk, axis=0)
            
            if background_depth is None:
                background_depth = chunk_max
            else:
                np.maximum(background_depth, chunk_max, out=background_depth)
            
    base_name = os.path.splitext(h5_local_path)[0]
    npy_path = f"{base_name}_bg.npy"
    png_path = f"{base_name}_bg.png"
    
    np.save(npy_path, background_depth)
    plt.imsave(png_path, background_depth, cmap='plasma')
    
    os.remove(h5_local_path)
    return f"Processed & Cleaned: {os.path.basename(h5_local_path)}"

def process_single_file(remote_file, fs):
    """Worker function to handle a single file from start to finish"""
    rel_path = remote_file.replace(f"{REPO_ID}/", "")
    local_path = os.path.join(LOCAL_BASE_DIR, rel_path)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # 1. Handle Depth Maps
    if "/depth_maps/" in remote_file and remote_file.endswith(".h5"):
        base_name = os.path.splitext(local_path)[0]
        if os.path.exists(f"{base_name}_bg.npy"):
            return f"Skipped (Already Processed): {rel_path}"
            
        fs.get_file(remote_file, local_path)
        result_msg = extract_and_cleanup_depth(local_path)
        return result_msg
        
    # 2. Handle Videos and Metadata
    else:
        if os.path.exists(local_path):
            return f"Skipped (Already Downloaded): {rel_path}"
            
        fs.get_file(remote_file, local_path)
        return f"Downloaded: {rel_path}"

def main():
    fs = HfFileSystem()
    print(f"Scanning Hugging Face repository: {REMOTE_BASE_DIR}...")
    
    all_files = fs.find(REMOTE_BASE_DIR, detail=False)
    total_files = len(all_files)
    print(f"Found {total_files} files. Starting parallel processing with {MAX_WORKERS} workers...\n")
    
    # Use ThreadPoolExecutor to run downloads and processing concurrently
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks to the pool
        futures = {executor.submit(process_single_file, f, fs): f for f in all_files}
        
        # As each file finishes, print its status
        for i, future in enumerate(as_completed(futures)):
            try:
                result = future.result()
                print(f"[{i+1}/{total_files}] {result}")
            except Exception as e:
                file_that_failed = futures[future]
                print(f"[{i+1}/{total_files}] ERROR on {file_that_failed}: {e}")

    print("\nAll files processed successfully!")

if __name__ == "__main__":
    main()