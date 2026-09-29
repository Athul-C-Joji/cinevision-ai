# cinevision-ai# CineVision AI

Cinematography analysis from video: shot size, framing, camera angle, lens size, lighting, composition and camera movement, plus a shot-by-shot breakdown and an LLM-based shot planner.

Built as a portfolio project (MSc Big Data Analytics). Trained on a GTX 1650 (4 GB) and Kaggle's free T4 GPU.

<!-- Add a screenshot of the demo here once you have one, for example: ![Demo](docs/demo.png) -->

## What is in it

| Module | What it does | Trained? |
|---|---|---|
| **1. Video Analyzer** | Static labels from one frame (7 dimensions) and camera movement from 16 frames | Yes: frozen CLIP ViT-B/32 with 7 small linear heads (about 26.7k trainable parameters), plus a small LSTM for movement |
| **2. Script Generator** | Splits a video into shots (PySceneDetect), runs Module 1 on each shot, writes a shot-by-shot breakdown | No new training |
| **3. Script Planner** | Script text in, suggested shots per scene out, using the same label vocabulary | **No. This is LLM prompting (Gemini free tier), not a trained model, and it has not been evaluated** |
| **Demo app** | Streamlit page: upload a video, see the breakdown per shot | Uses Modules 1 and 2 |

The deep learning work is Modules 1 and 2. Module 3 is orchestration around an LLM.

## Results: ShotBench (held-out, 3,572 questions)

Each question has 4 options (random guessing is about 25%). The model scores every option from its own probabilities and picks the highest. Options that list several tags are scored as the average of the tag probabilities (my own design choice).

| Category | Questions | Accuracy |
|---|---|---|
| Shot size | 485 | 77.1% |
| Shot framing | 445 | 74.2% |
| Camera angle | 455 | 65.5% |
| Lighting type | 405 | 59.3% |
| Composition | 479 | 53.2% |
| Lighting | 350 | 52.9% |
| Lens size | 489 | 50.7% |
| Camera movement | 464 | 45.8% |
| **Average across the 8 categories** | 3,572 | **59.83%** |

For reference, the published baselines are GPT-4o at 59.3% and ShotVL-7B at 70.1%. The comparison is approximate, because the scoring methods differ. The gap to GPT-4o is about half a point, which is within noise, so read it as "comparable to", not "better than". ShotVL-7B is a fully fine-tuned model, while this one is a frozen CLIP with small heads.

**Caveats**
- ShotBench lists no film names, so overlap with the training data (ShotQA) cannot be fully ruled out. A filename check found no shared files, and a CLIP similarity check on the image questions found no sign that near-duplicate frames inflate the score. It cannot rule out other frames from the same films, and it did not cover the movement videos.
- Lens size, lighting and camera movement are the hardest dimensions for published models too.

Full details, design choices and the other experiments (per-class threshold tuning, a class-weighting experiment that did not help) are in `PROJECT_NOTES.md`, sections 11 to 16.

## Known limits

- Both models were trained only on ShotQA (professional film frames). Stock footage or phone video is out of domain, and predictions can look repetitive or wrong.
- Camera movement is shown as the model's top 2 ranked guesses, not confident detections. The movement model was trained on 1,038 clips.
- The Camera Angle vocabulary has 5 classes (Aerial, Dutch angle, High angle, Low angle, Overhead) and **no eye-level class**, so the Video Analyzer always names one of those five.
- A whole video is treated as one scene. There is no scene grouping.
- The Script Planner plans each scene on its own, its output varies from run to run, and there is no ground truth to measure it against.

## Data and licenses

- Training data: ShotQA (`Vchitect/ShotQA` on Hugging Face, gated, licensed CC BY-NC-ND 4.0, so non-commercial use only). Evaluation: ShotBench (`Vchitect/ShotBench`).
- No dataset files or trained weights are included in this repository. Checkpoints are gitignored, so a fresh clone cannot run inference until the models are trained (see `PROJECT_NOTES.md`, sections 10, 11 and 13).

## Setup (Windows, PowerShell, Python 3.11)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
# GPU build of PyTorch first (see the comments at the top of requirements.txt), then:
pip install -r requirements.txt
```

Create a `.env` file in the project root (it is gitignored) with:

```text
HF_TOKEN=your-hugging-face-token
GEMINI_API_KEY=your-gemini-api-key
```

`HF_TOKEN` is needed to download the gated datasets, and `GEMINI_API_KEY` only for the Script Planner.

## Usage

```powershell
# Analyze a video (per-shot labels)
python -m src.inference.video_analyzer path\to\video.mp4

# Shot-by-shot text breakdown
python -m src.inference.script_generator path\to\video.mp4

# Demo app
python -m streamlit run app\streamlit_app.py

# Script Planner (calls the Gemini API)
python -m src.script_planner.planner data\scripts\sample_script.txt

# ShotBench evaluation (download the data first with download_shotbench.py)
python -m src.training.eval_shotbench
```

## Repository layout

```text
src/data/            label vocabularies, dataset code, splits
src/models/          static classifier (frozen CLIP + heads), movement LSTM
src/training/        training, threshold tuning, ShotBench evaluation, overlap check
src/inference/       shot segmentation, video analyzer, script generator
src/script_planner/  LLM-based shot planner
app/                 Streamlit demo
reports/             evaluation outputs
PROJECT_NOTES.md     full project log: decisions, experiments, results, caveats
```