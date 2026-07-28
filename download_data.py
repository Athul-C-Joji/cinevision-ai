from huggingface_hub import hf_hub_download

path = hf_hub_download(
    "Vchitect/ShotQA",
    "data.tar.gz",
    repo_type="dataset",
    local_dir="data/raw",
)
print("Downloaded to:", path)