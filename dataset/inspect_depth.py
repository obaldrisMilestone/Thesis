import h5py
import numpy as np
from huggingface_hub import HfFileSystem

REPO = "datasets/nvidia/PhysicalAI-SmartSpaces"
DEPTH_DIR = f"{REPO}/MTMC_Tracking_2025/val/Hospital_000/depth_maps"

fs = HfFileSystem()
files = fs.ls(DEPTH_DIR)

if files:
    # Extract the string path from the metadata dictionary
    depth_file = files[0]["name"] 
    
    local_path = "/home/user/thesis/code/dataset/depth_map.h5"
    print("downloading depth map file...")
    fs.get_file(depth_file, local_path)
    print(f"Downloaded: {depth_file} to {local_path}")
    
    with h5py.File(local_path, 'r') as f:
        print(f"Keys: {list(f.keys())}")
        for key in f.keys():
            print(f"Shape of {key}: {f[key].shape}")