import h5py
import numpy as np
from huggingface_hub import HfFileSystem

REPO = "datasets/nvidia/PhysicalAI-SmartSpaces"
DEPTH_DIR = f"{REPO}/MTMC_Tracking_2025/val/Hospital_000/depth_maps"

def inspect_h5_structure(h5file, indent=0):
    for key in h5file.keys():
        item = h5file[key]
        prefix = "  " * indent
        if isinstance(item, h5py.Dataset):
            print(f"{prefix}[dataset] {key}: shape={item.shape}, dtype={item.dtype}")
        elif isinstance(item, h5py.Group):
            print(f"{prefix}[group]   {key}/")
            inspect_h5_structure(item, indent + 1)

if __name__ == "__main__":
    fs = HfFileSystem()

    print("Listing depth map files...")
    files = fs.ls(DEPTH_DIR, detail=True)
    files = [f for f in files if f["name"].endswith(".h5")]
    files.sort(key=lambda f: f["name"])

    for f in files:
        size_gb = f.get("size", 0) / 1e9
        print(f"  {f['name'].split('/')[-1]}  ({size_gb:.2f} GB)")

    if not files:
        print("No .h5 files found. Check the path.")
        exit(1)

    first_file = files[0]["name"]
    print(f"\nOpening first file via streaming: {first_file.split('/')[-1]}")

    with fs.open(first_file, "rb") as fobj:
        with h5py.File(fobj, "r") as h5f:
            print("\n=== Top-level structure ===")
            inspect_h5_structure(h5f)

            print("\n=== First dataset sample ===")
            def show_first_dataset(h5file):
                for key in h5file.keys():
                    item = h5file[key]
                    if isinstance(item, h5py.Dataset):
                        if len(item.shape) >= 2:
                            print(f"  Reading first frame of '{key}'...")
                            frame = item[0] if len(item.shape) == 3 else item[()]
                            print(f"  Shape: {frame.shape}")
                            print(f"  dtype: {frame.dtype}")
                            print(f"  min={np.min(frame):.4f}  max={np.max(frame):.4f}  mean={np.mean(frame):.4f}")
                        return
                    elif isinstance(item, h5py.Group):
                        show_first_dataset(item)
                        return
            show_first_dataset(h5f)

            print("\n=== File-level attributes ===")
            for k, v in h5f.attrs.items():
                print(f"  {k}: {v}")
