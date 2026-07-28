# CineVision AI — Project Brief

**Owner:** Athul | MSc Big Data Analytics | Target: portfolio piece for Sept 2026 placements
**Timeline:** 3–4 weeks
**Compute:** GTX 1650 (4GB, local prototyping) + Kaggle T4/P100 free tier (main training)

---

## 1. What the project is

An end-to-end cinematography analysis system with three modules, in priority order:

| # | Module | What it does | Is it trained? | Depends on |
|---|--------|--------------|-----------------|------------|
| 1 | **Video Analyzer** | Video → shot size, framing, camera angle, lens size, lighting type, lighting condition, composition, camera movement | ✅ YES — this is the core deep learning work | ShotQA training data |
| 2 | **Script Generator** | Video → formatted shot-by-shot breakdown (scene 1, shot 1: medium shot, eye-level...) | Built on Module 1 + shot segmentation (no new training) | Module 1's trained model |
| 3 | **Script Planner** | Script text → suggested shots for unfilmed scenes | LLM prompting + optional retrieval, NOT trained | Optionally grounded by Module 1's learned vocabulary |

**Build order: Module 1 → Module 2 → Module 3.** Module 1 is the only one that proves deep learning skill — it gets full time and priority. Modules 2–3 are comparatively cheap once it exists.

---

## 2. Architecture (Module 1 — the core)

Video → Frame sampler → CLIP ViT (FROZEN) → shared trunk → 7 static heads
↓ (shot size, framing, angle,
Frame embedding sequence lens, lighting type,
↓ lighting condition, composition)
Transformer/LSTM (small, trained)
↓
Camera movement head


- CLIP backbone stays **frozen** — you only train lightweight heads on top. This is what makes local GPU (4GB) training feasible at all.
- 7 dimensions are classified from a single representative frame.
- Camera movement needs a frame *sequence* (8–16 frames) → small temporal model.

**MVP baseline (no training, works day 1):** CLIP zero-shot classification against text prompts per dimension, + optical-flow heuristic for movement. Useful to demo something working immediately while the trained version is built. Swap in trained heads later — same interface.

### 2b. Label strategy — multi-label, not single-label

Confirmed via `meta.jsonl` analysis (61,405 rows, 709 films):
- **Frame Size** (5.0% multi-tagged, 7 classes), **Lens Size** (3.3%, 5 classes), **Composition** (10.3%, 6 classes) — nearly single-label in practice
- **Shot Type** (48.4% multi-tagged, 12 classes), **Lighting Type** (51.3%, 12 classes), **Lighting** (61.2%, 10 classes) — genuinely multi-label; tags describe layered/co-occurring properties, not mutually exclusive categories

**Decision:** multi-label classification for all 7 heads (sigmoid + `BCEWithLogitsLoss`, per-class threshold at inference), not single-label softmax. Chosen deliberately over the simpler single-label MVP after reviewing tradeoffs — more faithful to the data, at the cost of messier evaluation (per-class F1 instead of clean confusion matrices) and needing to separately compute a single-label-equivalent accuracy later if comparing directly to GPT-4o (59.3%) / ShotVL-7B (70.1%) baselines, which are likely single-answer scored.

Class vocabularies and multi-hot encoding logic live in `src/data/label_encoding.py` (`CLASS_VOCAB` dict + `encode_multihot()` function) — tested and working.

---

## 3. Datasets — priority order

| Priority | Dataset | Covers | Size/access notes |
|----------|---------|--------|--------------------|
| **1 (primary)** | **ShotQA** (`Vchitect/ShotQA`, HF, gated) | All 8 dimensions | ~70k QA pairs, ~50GB — **do NOT bulk download**, sample/stream selectively |
| **2 (eval)** | **ShotBench** (`Vchitect/ShotBench`, HF, gated) | All 8 dimensions | ~3.5k expert-annotated pairs, small — download in full. Use to compare against published baselines (GPT-4o 59.3% avg, ShotVL-7B 70.1% avg) |
| **3 (optional supplement)** | **CineScale** | Shot scale (9 classes) | 792K frames, 124 films. CSV annotations free via Mendeley (DOI `10.17632/th46h4vdwd.1`); actual frame images need manual request via cinescale.github.io — **start this request early, it's slow** |
| **4 (optional supplement)** | **MovieShots / MovieNet** | Shot size + movement | Use only if ShotQA's movement coverage proves thin |
| **5 (optional supplement)** | **FullShots** | Movement (8 classes), scale (6 classes) | github.com/litchiar/ShotClassification |

**Do not scrape** (shot.cafe, etc.) — copyright/ToS risk, and unnecessary since the above already covers everything needed. shot.cafe is fine as a manual visual reference tool only.

**ShotQA vs ShotBench:** same paper/authors, same 8 dimensions. ShotQA = train on it. ShotBench = evaluate on it. Keep separate — check for film overlap before trusting eval numbers.

**Note on ShotQA structure (confirmed via `list_repo_files`):** the repo contains only 7 files — `.gitattributes`, `README.md`, `data.tar.gz` (~40GB, all images), `meta.jsonl` (~90MB, raw labels), `sft.json` (~24.7MB), `sft_v1.1.json` (~567MB), `grpo.json` (~4MB). The "sample/stream selectively" advice above doesn't apply cleanly here — `data.tar.gz` is a single non-seekable gzip archive, so there's no way to pull a subset without reading through the whole compressed stream anyway. Since disk space isn't a constraint (600GB+ free), the practical approach is: download the full tar once, extract locally, then subsample as needed for local GTX 1650 prototyping vs. the full set for Kaggle training.

**`meta.jsonl` schema:** each row has `img_id` (filename, e.g. `NJ8E6FEM.jpg`) plus rich ShotDeck-sourced metadata — `Frame Size`, `Shot Type`, `Lens Size`, `Composition`, `Lighting`, `Lighting Type` map almost directly to the 7 target dimensions. Also includes `Director`, `Cinematographer`, `Camera`, `Lens`, `Story Location`, etc. (not used for classification, but useful context). Field completeness is high for the dimensions that matter (97–99% non-null); `Actors`, `VFX`, `Notes`, `Film Stock / Resolution` are sparse and not relevant here. License is `cc-by-nc-nd-4.0` — fine for academic/portfolio use, would need separate permission for any future commercial use.

---

## 4. Repo structure

cinevision-ai/
├── data/{raw,processed,scripts}/
├── notebooks/ # 01_data_exploration → 06_evaluation
├── src/
│ ├── data/ # dataset.py, download.py, preprocessing.py, label_encoding.py, split_films.py
│ ├── models/ # static_classifier.py, movement_classifier.py, backbone.py
│ ├── training/ # train.py, evaluate.py
│ ├── inference/ # video_analyzer.py, shot_segmentation.py, script_generator.py
│ └── script_planner/ # prompt_templates.py, retrieval.py
├── app/ # streamlit_app.py
├── checkpoints/ # gitignored
├── reports/
├── requirements.txt
├── PROJECT_NOTES.md # this file
└── .gitignore


---

## 5. Weekly plan

- **Week 1:** HF access approved → inspect ShotQA schema, class balance, image-vs-video split → stratified sample (not full 50GB) → PyTorch Dataset class
- **Week 2:** Train the 7 static-dimension multi-task classifier (frozen CLIP + heads) → evaluate on ShotBench, confusion matrices
- **Week 3:** Temporal movement model → shot segmentation (PySceneDetect) → wire up Video Analyzer + Script Generator pipelines
- **Week 4:** Script Planner (LLM + optional retrieval, bonus only if on schedule) → Streamlit polish → Grad-CAM/explainability → write-up comparing to GPT-4o/ShotVL baselines

---

## 6. Compute strategy

- **Local (GTX 1650, 4GB):** prototyping, debugging, small-scale smoke tests. Use `torch.cuda.amp` mixed precision, small batch sizes (8–16), keep CLIP frozen.
- **Kaggle (free T4/P100, 16GB, 30 GPU-hrs/week):** the real training runs, once code is confirmed working locally. Needs phone verification for GPU + internet access. **Stop sessions when not actively in use — idle sessions still burn the weekly quota.**
- **Colab:** backup if Kaggle queue is busy.
- Checkpoint model weights regularly — free-tier sessions disconnect.

---

## 7. Beginner watchlist (don't skip these)

- Split train/val/test **by film**, not by frame — same-scene frames leak between splits otherwise. (Done — see section 10, `data/processed/film_splits.csv`.)
- Check class balance before training — "medium shot" will dominate if unaddressed. (Checked — see section 2b, less severe than expected but confirmed multi-label.)
- Look at confusion matrices per dimension, not just overall accuracy. (Note: with multi-label, per-class precision/recall/F1 replaces standard confusion matrices — plan for this in evaluation code.)
- Pin package versions in `requirements.txt` early (Kaggle vs local mismatches are common).
- Never commit secrets (HF tokens, API keys) — use `.env`, gitignored.
- Never commit large files (checkpoints, videos, raw data) to git.

---

## 8. Tools

- **VS Code** for all local code editing.
- **Claude Code extension** (Extensions → search "Claude Code") for in-editor AI help with full repo context — needs Pro plan or API key.
- **GitHub CLI (`gh`)** authenticated via `gh auth login` — enables git push/pull and Claude Code git actions without manual token entry.
- **GitHub repo** `Athul-C-Joji/cinevision-ai` — `.gitignore` covers `data/raw/`, `data/processed/` (partially — see note below), `checkpoints/`, `*.mp4`, `.env`, `__pycache__/`.
  - Note: `data/processed/film_splits.csv` is small and useful to version — confirm it's not being blanket-ignored if `data/processed/` is fully excluded; adjust `.gitignore` to allow this specific file if needed.
- **Hugging Face** — authenticated via `hf auth login` (CLI renamed from `huggingface-cli` to `hf`). Token stored in `.env` as `HF_TOKEN` (never committed).

---

## 9. Honesty notes for interviews/report

- Be explicit: **Video Analyzer + Script Generator = your deep learning contribution.** **Script Planner = LLM orchestration**, not trained. Don't blur this — interviewers ask, and clarity reads as maturity.
- Lens size, lighting condition, camera movement are the hardest dimensions across nearly every published model (including GPT-4o, ShotVL) — don't be surprised or embarrassed if your model struggles there too; it's a documented, known-hard problem, not a modeling mistake.
- The multi-label vs single-label decision (section 2b) is itself a good talking point — shows deliberate tradeoff analysis rather than defaulting to the simplest option.

---

## 10. Current status / next actions

**Done (Week 1, Day 1):**
- Environment fully set up: Python 3.11, Git, GitHub repo, VS Code, Claude Code extension, GitHub CLI authenticated, venv active, `requirements.txt` installed (fixed `pyscenedetect` → `scenedetect` typo)
- Kaggle: Tesla T4 GPU confirmed working (`torch.cuda.is_available()` → `True`), phone verified
- Hugging Face: authenticated, confirmed access to `Vchitect/ShotQA` (no separate request needed)
- `meta.jsonl` inspected: 61,405 rows, 709 unique films, schema mapped to target dimensions (see section 3)
- Multi-label decision made and implemented (section 2b) — `src/data/label_encoding.py` built and tested
- Film-level 70/15/15 split done — `src/data/split_films.py`, seed=42, saved to `data/processed/film_splits.csv` (train 44,218 / val 8,430 / test 8,757 rows)
- `data.tar.gz` (~40GB) download in progress via `download_data.py`

**Next actions:**
1. Extract the tar (`extract_data.py` ready, not yet run)
2. Confirm extracted folder structure maps `img_id` → actual file path
3. Decide: upload full extracted dataset as a Kaggle Dataset (avoid re-downloading 40GB every session) vs. re-download in-notebook
4. Build the PyTorch `Dataset` class using `label_encoding.py` + `film_splits.csv` + extracted images
5. Inspect `sft.json` / `sft_v1.1.json` / `grpo.json` schema (the QA-formatted files — not yet looked at)
6. Check ShotBench's exact schema/eval format for fair baseline comparison
7. Verify `torch.cuda.is_available()` returns `True` **locally** on the GTX 1650 (not yet confirmed — only confirmed on Kaggle so far); reinstall torch with correct CUDA index URL if not