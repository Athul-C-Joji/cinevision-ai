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


---

## 12. Module 1 movement sub-pipeline — threshold tuning results (Week 3)

**Per-class threshold tuning done** for the LSTM movement classifier (`src/training/find_movement_thresholds.py`, mirroring `find_best_thresholds.py`'s sweep-and-pick-max-F1 approach, thresholds 0.05–0.95). Same dramatic improvement pattern as Module 1's static classifier:

- Macro-F1 flat 0.5: **0.0088**
- Macro-F1 tuned (all 21 classes): **0.1870**
- Macro-F1 tuned (15 classes with actual val support): **0.2617** ← the number to cite

**Caveat worth stating explicitly (don't let this slide past unnoticed):** 6 of the 21 classes — Push out, Pull in, Trucking left, Trucking right, Zoom in, Zoom out — have **zero positive examples in the val split**. Their F1 is mathematically pinned at 0.0 regardless of threshold, and the "tuned" threshold values saved for them (all 0.05, the grid minimum) are meaningless artifacts of no signal, not real tuning. This isn't a script bug — it's the direct consequence of some movement classes having as few as 3–15 examples in the *entire* dataset (documented in section on movement label normalization), so a video-level 70/15/15 split can easily leave a rare class with zero representation in val or test.

Best-performing tuned class: **Push in** (58 val support) at **0.4813 F1**. Weakest genuinely-evaluable classes (Arc=4, Camera roll=3, Rack focus=4 support) stayed low even after tuning — consistent with the same data-scarcity conclusion as section 11's class-weighting experiment.

**Files added:** `src/training/find_movement_thresholds.py`
**Outputs:** `reports/movement_best_thresholds.json`, `reports/movement_eval_metrics_tuned.csv`

**Next:** shot segmentation via PySceneDetect, then Module 2 (Script Generator) pipeline wiring, per Week 3 plan.

## 13. Movement data pipeline, Module 2 build, and domain-shift test (Week 3)

**Movement data prep (Module 1, movement sub-model):**
- The 1,058 video-named `.mp4` files in `data/raw/images/` are the movement training clips. Labels come from the `"videos"` key in `grpo.json` (multiple-choice movement questions with an `<answer>` tag). All 1,058 files were verified present on disk and all labels parsed.
- Raw labels are free text: about 15-18 atomic camera-movement primitives combined with "and" (simultaneous) or "Firstly X, then Y" (sequential), with inconsistent punctuation. There were 309 distinct raw strings.
- `src/data/movement_label_encoding.py` parses them clause by clause into a 21-class multi-hot vocabulary (`MOVEMENT_CLASS_VOCAB`). First run: 0 unmatched rows, 0 unmatched clauses. Output: `data/processed/movement_labels_encoded.csv`.
- Bug fixed in `src/data/extract_movement_labels.py`: it wrote CSV rows with manual string formatting, so labels containing commas broke parsing. Now uses `csv.writer` with `newline=""`.
- `src/data/extract_movement_frames.py` decodes each clip in memory and saves 16 evenly spaced frames per clip under `data/processed/movement_frames/<clip_id>/`. 1,038 of 1,058 clips succeeded. 20 failed with "moov atom not found" (16 distinct corrupted source videos) and were excluded, consistent with the broken-images finding in section 10.
- Split: by source video, not by clip (see section 15 for the verified counts: 711 / 183 / 144 clips).
- `MovementDataset` (`src/data/movement_dataset.py`) mirrors `ShotDataset`, returning (16-frame sequence, multi-hot label).

**Movement model choice (LSTM vs Transformer):** `MovementClassifier` (`src/models/movement_classifier.py`) supports both heads on frozen CLIP frame embeddings. LSTM head: 331,413 trainable params. Transformer head: 1,203,349. Both trained 5 epochs (`src/training/train_movement.py`), compared on val macro average-precision (flat-0.5 macro-F1 read 0.0000 for both, the same threshold-suppression problem as section 11).
- LSTM: steady, plateaued (last 3 epochs 0.1943 / 0.1950 / 0.1942).
- Transformer: higher peak at epoch 3 (0.2319) but noisy (last 3 epochs 0.2319 / 0.1915 / 0.1948) while train loss kept falling, a sign of overfitting on 711 training clips.
- **Decision:** kept the LSTM. Checkpoints for both are saved (`checkpoints/movement_lstm.pt`, `checkpoints/movement_transformer.pt`).

**Module 2 build (shot segmentation, Video Analyzer, Script Generator):**
- `src/inference/shot_segmentation.py` uses PySceneDetect `ContentDetector`. `src/inference/video_analyzer.py` runs each shot's representative frame through the static classifier and each shot's 16 sampled frames through the movement LSTM. `src/inference/script_generator.py` formats the result as a shot-by-shot breakdown, saved to `data/scripts/`.
- PySceneDetect finds cuts, not scene groupings, so the Script Generator treats the whole video as one scene containing all detected shots. Real scene grouping has not been built.

**Domain-shift test (informal, not a benchmark):** ran the full pipeline on a downloaded stock b-roll clip that is not part of ShotQA.
- Shot detection worked: 4 real cuts found, and saved per-shot frames were visibly different and correctly indexed.
- Predictions, however, were nearly identical across all 4 shots regardless of content, including contradictory movement labels (for example Tilt up + Tilt down).
- **Hypothesis (not tested):** domain shift. Both models were trained only on professional film footage from ShotQA, and generic stock footage is out of domain. This was not confirmed by a controlled experiment.
- The contradictory movement labels were later traced to a separate cause, the threshold over-triggering described in section 14, so part of what looked like domain shift may have been that bug.
- The model's real evaluation is the held-out ShotQA test split and ShotBench, not informal stock footage.

## 14. Movement classifier inference fix — threshold over-triggering (Week 3)

**Problem discovered through real-data testing (not the stock-footage domain-shift test):** ran the full pipeline on several real ShotQA clips with known ground truth and compared predictions directly. Results were bad in a specific, diagnosable way — the movement model predicted 6-8 labels per clip regardless of actual content, including physically impossible simultaneous opposite pairs (Push in + Pull out, Tilt up + Tilt down, Boom up + Boom down all firing together on the same clip).

**Root cause:** inspected `movement_best_thresholds.json` (from section 12's tuning) — every single one of the 21 tuned thresholds was at or near the sweep floor (0.05-0.15, none above 0.15). This happened because F1-maximization was run on a small, sparse val split (183 clips across 21 classes, many with single-digit support), so the optimizer kept walking toward the minimum threshold rather than settling on a genuinely confident cutoff.

**Diagnosis confirmed by inspecting raw probabilities directly** (bypassing thresholds entirely): across several real test clips, the highest probability for any class never exceeded ~0.35 — well below any threshold that would represent real confidence. But critically, the *relative ranking* was often meaningful: the single highest-probability class matched the true label on 2 of 4 manually-checked real clips, and was on the correct axis (panning) on a 3rd. This showed the model had learned real, weak signal — the failure was in how that signal was being turned into discrete predictions, not in the model itself.

**Fix:** switched `classify_movement()` in `src/inference/video_analyzer.py` from absolute-threshold selection to **top-2 selection** (always return the 2 highest-probability classes, regardless of their raw score). Result: 2 labels per clip instead of 6-8, zero contradictory opposite-pairs, same underlying accuracy on the small manual test (~50% top-1 correct) but dramatically more usable and honest output.

**Framing for report/interviews:** this is a deliberate, evidence-driven design decision — threshold-based multi-label inference assumes well-separated per-class confidence, which testing showed doesn't hold for the movement model (likely due to only 5 training epochs on 711 clips spread across 21 classes). The static classifier's thresholding, by contrast, genuinely works — its probabilities are better separated, and threshold tuning there produced real, defensible gains (section 11). Worth citing as an example of diagnosing a problem empirically rather than assuming a fix and moving on. `movement_best_thresholds.json` / threshold-based inference is effectively superseded for the movement model specifically; the static classifier is unaffected and still correctly uses its tuned thresholds.

**Files changed:** `src/inference/video_analyzer.py` (`classify_movement()` rewritten)
**Sample outputs regenerated:** `data/scripts/-4dDC0lPRB0.webm_14.txt`, `data/scripts/4005541-hd_1920_1080_30fps.txt`


---

## 15. Movement dataset — verified clip counts (Week 3)

Counted directly from `data/processed/movement_splits.csv` (columns: `filename`, `base_video_id`, `split`) with a throwaway counting script. These are the verified numbers. Earlier figures written in chat or memory were not checked and should not be reused.

| Split | Clips | Source videos (`base_video_id`) | Avg clips per video |
|---|---|---|---|
| Train | 711 | 90 | 7.9 |
| Val | 183 | 19 | 9.6 |
| Test | 144 | 20 | 7.2 |
| **Total** | **1,038** | **129** | 8.0 |

**How the split was made:** by source video, not by clip, so clips from the same video never appear in two different splits (same reason as the film-level split in section 10).

**Why the clip ratios are not exactly 70/15/15:** at video level the split is almost exactly 70/15/15 (90 / 19 / 20 of 129 videos). The clip percentages (68.5% / 17.6% / 13.9%) drift because videos contribute different numbers of clips.

**What this means for the results (honest caveat):**
- The val split is only 19 source videos and the test split only 20. Movement metrics are therefore noisy, and many of the 21 classes have single-digit support.
- This backs up sections 12 and 14: tuning thresholds on 183 clips from 19 videos is fragile, which is part of why the tuned thresholds collapsed to 0.05–0.15.
- For the report: state these counts alongside any movement result, and describe the movement model's numbers as indicative, not definitive.

---

## 16. Held-out evaluation on ShotBench (Week 3)

### 16.1 What was evaluated
- Benchmark: `Vchitect/ShotBench`, `test.tsv`, 3,572 multiple-choice questions, always 4 options (A-D), random guessing about 25%.
- 3,108 image questions (3,049 unique image files) covering the 7 static dimensions, and 464 video questions covering camera movement only.
- Models: the final Module 1 artifacts, unchanged. Static: frozen CLIP + 7 linear heads (`checkpoints/best_model.pt`). Movement: LSTM (`checkpoints/movement_lstm.pt`).
- No files failed to load (0 of 3,513).

### 16.2 How answers were scored (design choices)
- Each of the 4 options gets a score from the model's probabilities, and the highest score is the answer. No thresholds are used, so the val-tuned thresholds from section 11 do not affect these numbers.
- **Combination options** (e.g. "Aerial, Overhead") are scored as the AVERAGE of the model's probabilities for the tags inside the option. This is my own design choice, not something taken from the ShotBench paper.
- One wording difference is mapped: ShotBench says "Single" where my Shot Framing vocabulary says "Clean single".
- Movement options are free text, so they are parsed with the existing `src/data/movement_label_encoding.py` parser. This parser ignores the order of movements ("Firstly Pan left, then Tilt up" and the reverse give the same tags).
- An option with no matchable tags gets the lowest score and can never win. Ties give fractional credit.
- Code: `src/training/eval_shotbench.py`. Data download: `download_shotbench.py`. Existing modules were not modified.
- Outputs in `reports/`: `shotbench_results.csv`, `shotbench_single_vs_combo.csv`, `shotbench_predictions.csv`, `shotbench_summary.json`.

### 16.3 Results (full run, 3,572 questions)

| Category | Questions | Accuracy |
|---|---|---|
| Shot size | 485 | 0.7711 |
| Shot framing | 445 | 0.7416 |
| Camera angle | 455 | 0.6549 |
| Lighting type | 405 | 0.5926 |
| Composition | 479 | 0.5324 |
| Lighting | 350 | 0.5286 |
| Lens size | 489 | 0.5072 |
| Camera movement | 464 | 0.4580 |

- **Average across the 8 categories (macro): 0.5983. Average over all questions (micro): 0.5998.**
- Image questions only (7 static categories): 0.6210.
- For reference, the published baselines recorded in section 3: GPT-4o 59.3%, ShotVL-7B 70.1%.
- Camera movement, lens size and lighting are the lowest, matching the "known-hard dimensions" note in section 9.
- Single-tag vs combination correct answers, by category (accuracy on single / on combination):
  camera angle 0.6618 (n=414) / 0.5854 (n=41); camera movement 0.4704 (n=389) / 0.3933 (n=75); composition 0.5218 (n=458) / 0.7619 (n=21); lens size 0.4979 (n=478) / 0.9091 (n=11); lighting 0.4667 (n=240) / 0.6636 (n=110); lighting type 0.5333 (n=330) / 0.8533 (n=75); shot framing 0.7366 (n=391) / 0.7778 (n=54); shot size 0.7660 (n=470) / 0.9333 (n=15).
  Several combination groups are very small, so these are not reliable comparisons.
- Camera movement: 11 questions had a "Dolly zoom" option that my movement vocabulary has no class for; in 6 of them it was the correct answer, so those 6 could not be answered correctly. 1 movement question ended in a tie.

### 16.4 Leakage / overlap checks
- Filename check: 0 of 3,513 ShotBench files exist in `data/raw/images` (proves only that filenames differ; ShotBench lists no film names).
- Image similarity check (`src/training/check_shotbench_overlap.py`): frozen CLIP fingerprints, nearest ShotQA image (all 57,285 jpg/jpeg/png files searched) for each of the 3,049 ShotBench images. Median nearest similarity 0.8715; 685 images at or above 0.90, 39 at or above 0.95, 11 at or above 0.98, highest 0.9934.
- Accuracy on image questions by similarity group: below 0.85 = 0.6025 (n=893); 0.85-0.90 = 0.6290 (n=1,515); 0.90-0.95 = 0.6288 (n=660); 0.95 and above = 0.6000 (n=40). Accuracy does not rise for the most similar images, so there is no sign that near-duplicate frames inflate the score. The similarity cut-offs are arbitrary starting points.
- The two most similar pairs were opened and inspected by eye: similar-looking shots, not the same frame. The other very-close pairs were not inspected.
- 836 ShotQA images were unreadable in this script, versus 824 in `reports/broken_images.csv` (section 10). The 12 difference is unexplained and does not affect the conclusion.

### 16.5 Caveats (must be stated with the results)
- This check cannot rule out that ShotBench uses other frames from films in ShotQA (same film, different scene). Such overlap could give a small boost to every question equally, which the similarity groups would not reveal. Not tested.
- Movement videos were not covered by the similarity check.
- The comparison to published baselines is approximate. The scoring methods differ (published baselines answer from the question text; this model scores each option from tag probabilities). The gap to GPT-4o (about 0.5 points) is within rough sampling noise (about +/-1 point on the overall average, a rough binomial estimate), so the honest wording is "comparable to", not "better than".
- ShotVL-7B is a fully fine-tuned vision-language model, while this model is a frozen CLIP with about 26.7k trainable head parameters (static) and a small LSTM (movement). The 10-point gap to ShotVL-7B is unsurprising.
- Combination options are scored by averaging tag probabilities (a design choice). The movement parser ignores movement order.
- A 20-question-per-category smoke test earlier read 0.75 average; this was small-sample noise, and only the full-run numbers above are recorded.

### 16.6 Talking points
- Verified the result instead of taking a high-looking number at face value: a suspiciously good 20-question smoke test was followed by the full run and a leakage check.
- The overlap check found and fixed a bug in my own script (it originally searched `.jpg` files only and missed 420 images).
- Frozen CLIP + linear heads reach roughly GPT-4o-level on this benchmark, at a fraction of the cost, with a documented gap to a fully fine-tuned model.


---

## 17. Streamlit demo app and pinned requirements (Week 3)

### 17.1 What the app does
- `app/streamlit_app.py`: upload a video, click "Analyze video", and get a shot-by-shot breakdown. Each shot shows a thumbnail of the frame the static classifier saw, the 6 static labels plus composition, the top-2 movement guesses, and start/end timecodes. The full text breakdown can be downloaded as a `.txt` file.
- Run from the project root: `python -m streamlit run app\streamlit_app.py` (opens at `http://localhost:8501`).
- Confirmed working by the author on local runs. (Add here which test videos were used, if worth recording.)

### 17.2 How it is built
- Uses the same building blocks as `analyze_video()` in `src/inference/video_analyzer.py` (segment_video, decode_all_frames, classify_static, classify_movement), in the same order, and `format_shot_description()` from `script_generator.py` for the text layout. It does not call `analyze_video()` directly, so that it can keep a per-shot thumbnail and load the models once (cached with `st.cache_resource`).
- **Maintenance note:** this duplicates a small amount of pipeline logic. If `analyze_video()` changes, `app/streamlit_app.py` needs the same change.
- No existing modules in `src/` were modified. Uploaded videos are written to a temporary folder and deleted after analysis, not saved in the project.

### 17.3 Known limits (also shown in the app's sidebar)
- Both models were trained only on ShotQA (professional film frames), so other footage (stock clips, phone video) is out of domain, and predictions can look repetitive or wrong (see section 13).
- Movement shows the top 2 ranked guesses, not confident detections, and the two can contradict each other (see the movement threshold finding in section 13).
- The whole video is treated as one scene ("Scene 1"); there is no scene grouping.
- Videos over 1,200 frames are refused because every frame is decoded into memory. **The 1,200 figure is a cautious guess, not a measured limit.**
- The sidebar quotes the ShotBench results from section 16: 59.83% average across the 8 categories, camera movement weakest at 45.8%.

### 17.4 Pinned requirements
- `requirements.txt` now pins every package to the version installed in the working local venv (Python 3.11), for example torch 2.11.0, torchvision 0.26.0, transformers 5.14.1, streamlit 1.60.0, scenedetect 0.7.1.
- torch and torchvision are pinned without the `+cu128` build tag, because that tag only exists on PyTorch's own download server and breaks a plain `pip install` elsewhere. The file header explains how to install the CUDA build. `torchaudio` was left out (unused).
- Not yet tested: installing from this file on a clean machine or on Kaggle.


---

## 18. Script Planner (Module 3): LLM orchestration, not trained (Week 4)

### 18.1 What it does
- Script text in, suggested shots per scene out, using the same label vocabulary as Module 1 (the 7 static dimensions plus Movement).
- Files: `src/script_planner/prompt_templates.py` (prompt text, independent of which LLM is used) and `src/script_planner/planner.py`.
- Run: `python -m src.script_planner.planner data\scripts\sample_script.txt` (optional `--max-scenes N`, `--max-shots N`, `--model NAME`). Writes `data/scripts/<name>_plan.txt` and `.json`.

### 18.2 How it works
- The script is split into scenes by lines starting with INT. / EXT. (regex). Each scene is sent to the LLM on its own, together with a system prompt that lists every allowed label (built from `CLASS_VOCAB` and `MOVEMENT_CLASS_VOCAB`).
- The reply is JSON. Every label is checked against the vocabulary (exact match ignoring case). Labels that do not match are dropped and shown in the output. **This checks spelling only, not whether a suggestion is good.**
- Temporary API errors (429, 503, 500) are retried with growing waits (20s, 40s, 60s). `--max-scenes` (default 8) limits the number of requests.
- Nothing in `src/` outside `src/script_planner/` was modified. `retrieval.py` (planned in section 4) was not built.

### 18.3 Choice of LLM
- Uses the Gemini API free tier through the `google-genai` package (2.25.0), with `GEMINI_API_KEY` in `.env` (gitignored, never committed). A first version was written for the Anthropic API, but it needs paid credit, so it was replaced.
- Models tried: `gemini-2.5-flash` returned 404 ("no longer available to new users"); `gemini-3.8-flash` returned 503 "high demand" on every retry for both scenes; `gemini-3.5-flash-lite` worked and is the default.
- Model names, availability and free-tier limits change and vary per project (Google's own docs say so), so check `aistudio.google.com/rate-limit`.

### 18.4 Sample run
- Input: `data/scripts/sample_script.txt` (a 2-scene example written for testing). Output: 7 shots (4 in scene 1, 3 in scene 2), 1,244 input tokens and 893 output tokens in total, model `gemini-3.5-flash-lite`.
- The vocabulary check dropped two labels: "Natural light" (Lighting) and "Eye level" (Camera Angle). Neither is in the vocabulary. Possible reason for the second (hypothesis, not checked): eye level is the default angle and is not a class in the source data.

### 18.5 Limitations and observations (from two small runs)
- **Not evaluated.** There is no ground truth for "good" shot suggestions, so no accuracy number exists for this module.
- Output differs from run to run (a 1-scene test and the 2-scene run gave different labels for scene 1).
- Scenes are planned independently. Scene 2 is marked CONTINUOUS after a night scene but got "Daylight". Likely cause (hypothesis): the model never sees scene 1.
- Camera angle was "High angle" on 4 of 4 shots in the first run of scene 1 and 3 of 4 in the second. Possible default-label behavior (hypothesis).
- Mild additions beyond the script text (e.g. "anxiety", "precipice") despite the instruction not to invent.
- Script text is sent to Google. The free-tier data-use terms were not confirmed, so only non-sensitive scripts should be used.
- Google now recommends its newer Interactions API; this code uses `generate_content`, which worked in these runs and may need updating later.


### 18.7 Vocabulary check (supersedes the "unchecked" note in 18.4)
- Checked directly with `CLASS_VOCAB`: Camera Angle has 5 classes (Aerial, Dutch angle, High angle, Low angle, Overhead), so there is **no eye-level or neutral class**. Lighting has 10 classes (Backlight, Edge light, Hard light, High contrast, Low contrast, Side light, Silhouette, Soft light, Top light, Underlight), with no "natural light". This is why "Eye level" and "Natural light" were dropped by the validator in the sample run.
- Hypothesis (not tested): because the prompt asks for one label per dimension and every allowed angle is a marked angle, the planner is pushed toward "High angle" for ordinary shots. A possible fix, untried, is to let the prompt return an empty list when no label fits.
- Possible consequence for Module 1 (hypothesis, not measured): `classify_static` falls back to the top-1 class when no class crosses its threshold, so every shot gets some Camera Angle label and can never be reported as eye-level. Worth stating as a limitation of the label vocabulary in the write-up.


---

## 19. Streamlit app: Script Planner tab (Week 4)

### 19.1 What changed
- `app/streamlit_app.py` now has two tabs: **Video Analyzer** (the earlier code, unchanged in behaviour) and **Script Planner** (new). Committed as `67ce57e2`.
- The Script Planner tab calls `plan_script()` and `format_plan()` from `src/script_planner/planner.py`. Nothing in `src/` was modified.
- Inputs: an uploaded `.txt` script or pasted text, max scenes (default 4, up to 8, one API request per scene), max shots per scene, and the Gemini model name (defaults to the planner's own default).
- Output: scene-by-scene suggested shots with labels, any labels dropped by the vocabulary check, and `.txt` / `.json` downloads.
- The planner is imported inside the tab, so a planner problem (for example a missing package) does not break the Video Analyzer tab.
- `GEMINI_API_KEY` is read from `.env` (gitignored). The key is not shown in the app.

### 19.2 What was tested
- I ran the app locally and both tabs worked. No planner output or timings were recorded from this test, so this section makes no claim about the quality of the plans.

### 19.3 Limits (also shown in the app)
- Same as section 18: LLM-generated, not trained, not evaluated, varies between runs, scenes planned independently, script text sent to Google, spelling-only label check, no eye-level class in the Camera Angle vocabulary.
- The app duplicates a little pipeline logic from `analyze_video()` (see section 17.2), so changes to `analyze_video()` need the same change in the app.

---

## 20. Movement model: cached-embedding experiment and prior baseline (Week 4)

### 20.1 What was done
- `src/training/cache_movement_embeddings.py` runs frozen CLIP once over all clips and saves the 16x512 frame embeddings to `data/processed/movement_embeddings.pt` (gitignored). Shapes matched the section 15 clip counts: train (711, 16, 512), val (183, 16, 512), test (144, 16, 512).
- `src/training/train_movement_cached.py` trains the same kind of LSTM head on the cached embeddings, keeps the best epoch by val macro AP, and reports macro AP, top-1 and top-2 hit rate (how often the top-1 / at least one of the top-2 predicted classes is a true label). It never overwrites `checkpoints/movement_lstm.pt`; saved heads are not loadable by the current inference code.
- Hypothesis tested: frame-to-frame change in the embeddings carries camera-motion information. Three input modes: `embed` (as before), `diff` (differences only), `embed_diff` (both). 30 epochs, batch 32, 3 seeds each.

### 20.2 Results (val split, 183 clips from 19 videos; best epoch chosen on val)

| Mode | Seed 1 / 2 / 3 val macro AP | Top-1 hit (1/2/3) | Top-2 hit (1/2/3) |
|---|---|---|---|
| embed | 0.2282 / 0.2145 / 0.2277 | 0.295 / 0.317 / 0.301 | 0.448 / 0.464 / 0.454 |
| diff | 0.1451 / 0.1424 / 0.1717 | 0.240 / 0.317 / 0.317 | 0.372 / 0.443 / 0.410 |
| embed_diff | 0.2073 / 0.1892 / 0.2169 | 0.317 / 0.322 / 0.268 | 0.399 / 0.486 / 0.437 |

- Means over seeds (my arithmetic): macro AP embed 0.223, diff 0.153, embed_diff 0.204.
- Result: neither `diff` nor `embed_diff` beat plain `embed`. The frame-difference idea did not help here. Possible reason (hypothesis, not tested): the limit is the small training set (711 clips, 21 classes), not the input type.

### 20.3 No-model prior baseline (`src/training/movement_prior_baseline.py`)
- Ranks classes by how common they are in train and gives every val clip the same answer. Most common train classes: Push in 0.290, Tilt up 0.159, Pan left 0.149, Pan right 0.139, Static shot 0.134.
- Val: top-1 0.317, top-2 0.410, macro AP 0.1137 (15 classes with val support).
- Reading: the trained head's top-1 (about 0.30) is not better than always answering "Push in". Its top-2 (about 0.455) is slightly higher (about 8 clips of 183, not shown to be significant), and its macro AP (about 0.22, best epoch picked on val) is about double the prior's, which is the clearest sign of some learned signal.

### 20.4 Caveats
- Val is small and the best epoch was picked on it, so these numbers are optimistic and noisy (best epoch varied between 2 and 30). The test split has not been used for this experiment.
- The currently deployed `movement_lstm.pt` was not re-measured on this footing, so these runs are not shown to beat it. The earlier 0.194 was a last-epoch value from a different setup and is not comparable.
- The "about 50% top-1" remark in section 14 came from 4 clips and should not be quoted.


### 20.5 Deployed model measured on the same footing (`src/training/eval_deployed_movement.py`)
- Scores `checkpoints/movement_lstm.pt` (the model the app loads) on the val split, using the cached val embeddings and the same metric code as `train_movement_cached.py`. The head weights are loaded strictly. Only val was used; test is untouched.
- Sanity check: macro AP 0.1942 equals the last-epoch value recorded in section 12, so the model was loaded correctly.
- Val results (183 clips): deployed top-1 0.301, top-2 0.470, macro AP 0.1942. For comparison on the same split: no-model prior 0.317 / 0.410 / 0.1137; new `embed` heads (mean of 3 seeds, best epoch picked on val) about 0.304 / 0.455 / 0.223.
- Reading: the new recipe is not shown to beat the deployed model. Its higher AP comes with best-epoch selection on val, while the deployed model's number is a plain last-epoch value, and the deployed model's top-2 is slightly higher. Decision: keep `movement_lstm.pt`.
- All three top-1 values are close to always answering "Push in". The deployed model's top-2 is about 11 clips of 183 above the prior, which I would describe as weak signal, not shown to be significant.