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
                              ↓                             (shot size, framing, angle,
                    Frame embedding sequence                 lens, lighting type,
                              ↓                               lighting condition, composition)
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
│ ├── data/ # dataset.py, download.py, preprocessing.py, label_encoding.py, split_films.py, compute_pos_weights.py
│ ├── models/ # static_classifier.py, movement_classifier.py, backbone.py
│ ├── training/ # train.py, evaluate.py, find_best_thresholds.py
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
- **Week 2:** Train the 7 static-dimension multi-task classifier (frozen CLIP + heads) → evaluate on ShotBench, confusion matrices ✅ **DONE — see section 11**
- **Week 3 (current):** Temporal movement model → shot segmentation (PySceneDetect) → wire up Video Analyzer + Script Generator pipelines
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
- Look at confusion matrices per dimension, not just overall accuracy. (Note: with multi-label, per-class precision/recall/F1 replaces standard confusion matrices — implemented, see section 11.)
- Pin package versions in `requirements.txt` early (Kaggle vs local mismatches are common).
- Never commit secrets (HF tokens, API keys) — use `.env`, gitignored.
- Never commit large files (checkpoints, videos, raw data) to git. **Note: `.gitignore` initially only had the generic Python template and did NOT exclude these — fixed in section 11's session, verify this stays correct in future commits.**

---

## 8. Tools

- **VS Code** for all local code editing.
- **Claude Code extension** (Extensions → search "Claude Code") for in-editor AI help with full repo context — needs Pro plan or API key.
- **GitHub CLI (`gh`)** authenticated via `gh auth login` — enables git push/pull and Claude Code git actions without manual token entry.
- **GitHub repo** `Athul-C-Joji/cinevision-ai` — `.gitignore` covers `data/raw/`, `data/processed/*` (with `!data/processed/film_splits.csv` explicitly un-ignored), `checkpoints/`, `*.mp4`/`*.mkv`/`*.webm`/`*.mov`/`*.avi`, `.env`, `__pycache__/`.
- **Hugging Face** — authenticated via `hf auth login` (CLI renamed from `huggingface-cli` to `hf`). Token stored in `.env` as `HF_TOKEN` (never committed).

---

## 9. Honesty notes for interviews/report

- Be explicit: **Video Analyzer + Script Generator = your deep learning contribution.** **Script Planner = LLM orchestration**, not trained. Don't blur this — interviewers ask, and clarity reads as maturity.
- Lens size, lighting condition, camera movement are the hardest dimensions across nearly every published model (including GPT-4o, ShotVL) — don't be surprised or embarrassed if your model struggles there too; it's a documented, known-hard problem, not a modeling mistake.
- The multi-label vs single-label decision (section 2b) is itself a good talking point — shows deliberate tradeoff analysis rather than defaulting to the simplest option.
- **The class-weighting experiment (section 11) is another good talking point** — shows a hypothesis was tested (pos_weight should help rare classes), the result was honestly reported even though it didn't help, and the correct conclusion was drawn (data scarcity, not loss weighting, is the bottleneck) rather than just picking whichever number looked better.

---

## 10. Week 1 setup (historical)

**Done (Week 1, Day 1):**
- Environment fully set up: Python 3.11, Git, GitHub repo, VS Code, Claude Code extension, GitHub CLI authenticated, venv active, `requirements.txt` installed (fixed `pyscenedetect` → `scenedetect` typo)
- Kaggle: Tesla T4 GPU confirmed working (`torch.cuda.is_available()` → `True`), phone verified
- Hugging Face: authenticated, confirmed access to `Vchitect/ShotQA` (no separate request needed)
- `meta.jsonl` inspected: 61,405 rows, 709 unique films, schema mapped to target dimensions (see section 3)
- Multi-label decision made and implemented (section 2b) — `src/data/label_encoding.py` built and tested
- Film-level 70/15/15 split done — `src/data/split_films.py`, seed=42, saved to `data/processed/film_splits.csv` (train 44,218 / val 8,430 / test 8,757 rows)
- `data.tar.gz` (~40GB) downloaded and extracted — 58,343 files, verified 100% coverage against `meta.jsonl`, flat structure in `data/raw/images/`
- `validate_images.py` run: found 824 broken images (1.45%), saved to `reports/broken_images.csv`, auto-filtered by `ShotDataset`

---

## 11. Module 1 — training results & class-weighting experiment (Week 2)

**Architecture finalized as 7 dimensions, not 6:** `CLASS_VOCAB` in `src/data/label_encoding.py` was split — the original combined "Shot Type" field is now two separate heads, **Shot Framing** (7 classes) and **Camera Angle** (5 classes), both reading from the same raw `Shot Type` meta.jsonl field via a `SOURCE_FIELD` dict. Matches ShotBench's 8 core dimensions minus camera movement (handled separately by a not-yet-built temporal model). Full 7: Frame Size, Lens Size, Composition, Shot Framing, Camera Angle, Lighting Type, Lighting.

**Model built:** `src/models/static_classifier.py` — `StaticShotClassifier`, frozen CLIP (`openai/clip-vit-base-patch32`, ~151M frozen params) + 7 trainable linear heads (~26.7k trainable params). Had to bypass `self.clip.get_image_features()` due to a transformers-version API quirk (wrong return type); `forward()` calls `self.clip.vision_model()` + `self.clip.visual_projection()` directly instead, which is stable across versions.

**First full training run:** 5 epochs, frozen CLIP + 7 linear heads, unweighted `BCEWithLogitsLoss`. Loss decreased steadily every epoch (train 2.14→1.83, val 1.97→1.90), train/val gap small and stable — no overfitting, could likely train longer. `checkpoints/best_model_unweighted.pt`.

**Flat 0.5 threshold eval (`reports/eval_metrics.csv`)** revealed low recall on rare classes (e.g. HMI, LED, Tungsten all near-0 F1) despite reasonable precision — classic signature of a fixed threshold suppressing minority-class predictions.

**Per-class threshold tuning (`src/training/find_best_thresholds.py`)** — swept thresholds 0.05–0.95 per class on val split, picked whichever maximized each class's individual F1. Substantial improvement across every dimension:

| Dimension | Flat 0.5 | Tuned |
|---|---|---|
| Frame Size | 0.537 | 0.651 |
| Lens Size | 0.381 | 0.505 |
| Composition | 0.301 | 0.476 |
| Shot Framing | 0.586 | 0.660 |
| Camera Angle | 0.422 | 0.568 |
| Lighting Type | 0.319 | 0.423 |
| Lighting | 0.350 | 0.499 |

Genuinely data-starved classes (support <200, e.g. HMI=56, LED=76, Tungsten=183) stayed near-zero F1 even after tuning — a data scarcity problem, not a threshold problem.

**Note on methodology:** thresholds were tuned on the same val split used to report these numbers — some improvement is real generalization, some is overfitting to val noise (especially tiny-support classes). The true test will be re-running eval on the held-out **test** split or ShotBench once the pipeline reaches that stage, not re-checking val again.

**Class-weighted loss experiment:** computed per-class `pos_weight` (`src/data/compute_pos_weights.py`, raw weights ranged 0.99–82.63, capped at `MAX_POS_WEIGHT=20.0` to avoid instability), retrained 5 epochs (`checkpoints/best_model_weighted.pt`). Result: **no meaningful improvement over unweighted+tuned** — macro-F1 identical within noise on every dimension (all within ±0.003), and the target starved classes (HMI, LED, Tungsten) didn't meaningfully improve either. Conclusion: the bottleneck is feature separability for these classes given how little train data they have, not loss-function weighting — threshold tuning and pos_weight were correcting for the same underlying imbalance via different mechanisms, so combining them added nothing.

**Decision: kept the simpler unweighted model + tuned thresholds as the final Module 1 artifact.**
- `checkpoints/best_model.pt` (= restored copy of `best_model_unweighted.pt`)
- `reports/best_thresholds.json` (= restored copy of the unweighted-tuned thresholds)
- `reports/eval_metrics_tuned.csv` (final per-class P/R/F1, unweighted+tuned)

This finding is consistent with GPT-4o/ShotVL also struggling on similar hard categories (section 9) — worth citing directly in the report as a deliberate, documented experiment rather than an oversight.

**Files added this session:** `src/training/evaluate.py` (per-class P/R/F1 at a flat threshold), `src/training/find_best_thresholds.py` (threshold sweep + tuning), `src/data/compute_pos_weights.py` (pos_weight computation — used but ultimately not adopted for the final model).

**Bugs fixed:**
- `get_preprocess()` originally returned a local closure, unpicklable by Windows' `spawn`-based multiprocessing — blocked `num_workers>0`, forcing slow CPU-bound single-threaded data loading. Replaced with a module-level `ClipPreprocess` class in `static_classifier.py`.
- Missing `__init__.py` in `src/` and `src/data/` caused `ModuleNotFoundError` when running scripts via `python -m`. Added across all `src/` subpackages.
- `.gitignore` was just the generic Python template — did not actually exclude `data/raw/`, `checkpoints/`, or video files. Fixed by adding a project-specific block; verified with `git status` before committing that no dataset/video files get staged.

**Next:** Module 1 finalized — moving to shot segmentation (PySceneDetect) and Module 2 (Script Generator) per Week 3 plan. Also still pending: inspect `sft.json`/`sft_v1.1.json`/`grpo.json` schemas, check ShotBench's exact eval schema for a fair baseline comparison.