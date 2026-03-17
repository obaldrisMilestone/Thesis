from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="nvidia/PhysicalAI-SmartSpaces",
    repo_type="dataset",
    allow_patterns=["MTMC_Tracking_2025/val/Hospital_000/*"],
    ignore_patterns=["*.h5"], # This ignores the depth maps
    local_dir="./Hospital_000_Val"
)