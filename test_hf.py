from huggingface_hub import hf_hub_download

for fname in ["meta.jsonl", "sft.json", "sft_v1.1.json", "grpo.json"]:
    path = hf_hub_download("Vchitect/ShotQA", fname, repo_type="dataset")
    print(fname, "->", path)