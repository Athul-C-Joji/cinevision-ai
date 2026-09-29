import ast
import csv
import os
import sys
import tarfile
from collections import Counter

from huggingface_hub import hf_hub_download

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

REPO_ID = "Vchitect/ShotBench"
DEST = os.path.join("data", "raw", "shotbench")
os.makedirs(DEST, exist_ok=True)

# ---------- 1. Download ----------
for name in ["test.tsv", "images.tar", "videos.tar"]:
    print(f"Downloading {name} ...")
    hf_hub_download(
        repo_id=REPO_ID,
        filename=name,
        repo_type="dataset",
        local_dir=DEST,
    )
print("Download finished.")
print()

# ---------- 2. Extract ----------
for name in ["images.tar", "videos.tar"]:
    tar_path = os.path.join(DEST, name)
    marker = os.path.join(DEST, name + ".extracted")
    if os.path.exists(marker):
        print(f"{name} was already extracted, skipping.")
        continue
    print(f"Extracting {name} (this can take a few minutes) ...")
    with tarfile.open(tar_path) as tar:
        try:
            tar.extractall(DEST, filter="data")
        except TypeError:
            # older Python 3.11 builds do not know the "filter" option
            tar.extractall(DEST)
    with open(marker, "w") as fh:
        fh.write("done")
    print(f"Finished extracting {name}.")
print()

# ---------- 3. Check that every file listed in test.tsv exists ----------
tsv_path = os.path.join(DEST, "test.tsv")
with open(tsv_path, newline="", encoding="utf-8", errors="replace") as f:
    rows = list(csv.DictReader(f, delimiter="\t"))

found = Counter()
missing = Counter()
missing_examples = []

for r in rows:
    kind = ast.literal_eval(r["type"])[0]
    for p in ast.literal_eval(r["path"]):
        full = os.path.join(DEST, p)
        if os.path.exists(full):
            found[kind] += 1
        else:
            missing[kind] += 1
            if len(missing_examples) < 5:
                missing_examples.append(p)

print("=== Check against test.tsv ===")
print(f"Files found:   {dict(found)}")
print(f"Files missing: {dict(missing)}")
if missing_examples:
    print("Examples of missing paths:")
    for p in missing_examples:
        print(f"    {p}")

print()
print(f"=== Top level of {DEST} ===")
for entry in sorted(os.listdir(DEST)):
    full = os.path.join(DEST, entry)
    if os.path.isdir(full):
        inside = os.listdir(full)
        print(f"[folder] {entry}  ({len(inside)} items, first 3: {inside[:3]})")
    else:
        print(f"[file]   {entry}")