# Windows Setup & Execution Guide
## Affordance Prediction Pilot Study

This guide walks through setting up the environment and running the full ML pipeline on a Windows machine. Commands are for **Command Prompt** (`cmd`) or **PowerShell** — noted where they differ.

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Clone / Transfer the Repository](#2-clone--transfer-the-repository)
3. [Python Environment Setup](#3-python-environment-setup)
4. [Install Dependencies](#4-install-dependencies)
5. [Verify Data Files](#5-verify-data-files)
6. [Run the Pipeline](#6-run-the-pipeline)
7. [Check Outputs](#7-check-outputs)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. Prerequisites

Install these before anything else. Verify each with the check command.

### Python 3.11

Download from [python.org/downloads](https://www.python.org/downloads/) — pick **Python 3.11.x** (3.12 has LightGBM build issues on Windows).

During installation: **check "Add Python to PATH"**.

```cmd
python --version
:: Expected: Python 3.11.x
```

### Git

Download from [git-scm.com](https://git-scm.com/download/win).

```cmd
git --version
:: Expected: git version 2.x.x
```

### CUDA (if you have an NVIDIA GPU)

Check your GPU driver version first:

```cmd
nvidia-smi
:: Look for "CUDA Version: 12.x" in the top-right corner
```

Then download the matching CUDA Toolkit from [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads). CUDA 12.1 is recommended for PyTorch 2.3.

If you have no NVIDIA GPU, the CNN training will fall back to CPU (slow — hours per run). The LightGBM experiments do not require a GPU.

---

## 2. Clone / Transfer the Repository

**Option A — Git (if the repo is on GitHub):**

```cmd
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

**Option B — Copy from another machine:**

Copy the entire project folder to your Windows machine (e.g. via USB or network share). Then open a terminal in the root folder:

```cmd
cd C:\path\to\FINAL
```

Verify you can see the project layout:

```cmd
dir
:: Should show: project\  instructions.txt  WINDOWS_SETUP_GUIDE.md  etc.

dir project\
:: Should show: configs\  data\  outputs\  src\  requirements.txt  README.md
```

---

## 3. Python Environment Setup

Create an isolated virtual environment inside the repo root.

```cmd
python -m venv .venv
```

Activate it — **this must be done every time you open a new terminal**:

```cmd
:: Command Prompt:
.venv\Scripts\activate.bat

:: PowerShell:
.venv\Scripts\Activate.ps1
```

> **PowerShell note:** If you get a script execution error, run this once as Administrator:
> ```powershell
> Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
> ```

Confirm the environment is active — your prompt should show `(.venv)`:

```cmd
python --version
where python
:: Should point to .venv\Scripts\python.exe
```

Upgrade pip:

```cmd
python -m pip install --upgrade pip
```

---

## 4. Install Dependencies

### Step 4a — PyTorch (install first, separately)

Go to [pytorch.org/get-started/locally](https://pytorch.org/get-started/locally/) and select:
- PyTorch Build: **Stable**
- OS: **Windows**
- Package: **pip**
- Language: **Python**
- Compute Platform: **CUDA 12.1** (or **CPU** if no GPU)

Copy the generated command. It will look like one of these:

```cmd
:: CUDA 12.1:
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121

:: CPU only:
pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cpu
```

Verify:

```cmd
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
:: Expected: 2.3.1   CUDA: True  (or False if CPU-only)
```

### Step 4b — Core ML packages

```cmd
pip install lightgbm>=4.3.0 optuna>=3.6.0 shap>=0.45.0
pip install scikit-learn>=1.4.0 scipy>=1.13.0 pandas>=2.2.0 pyarrow>=16.0.0
pip install matplotlib>=3.9.0 seaborn>=0.13.0 Pillow>=10.3.0
pip install tqdm requests openpyxl
```

Verify the key packages:

```cmd
python -c "import lightgbm, optuna, shap, sklearn, scipy, pandas, matplotlib, seaborn; print('All OK')"
```

### Step 4c — numpy pinned version

The project requires `numpy<2` (LightGBM and the pinned transformers version both break on numpy 2.x):

```cmd
pip install "numpy>=1.26,<2"
python -c "import numpy; print(numpy.__version__)"
:: Expected: 1.26.x
```

### Step 4d — Packages NOT needed for the ML pipeline

The following are only required if you intend to re-run the VLM annotation or segmentation stages (which are already complete — all outputs are in `project/data/`). **Skip these unless you need to regenerate data:**

- `transformers`, `autoawq`, `qwen-vl-utils` (VLM inference)
- `detectron2` (Mask2Former segmentation — no Windows binary, requires WSL2 or Docker)
- `accelerate` (multi-GPU training)

---

## 5. Verify Data Files

Before running any training, confirm that all input data is present. Run these checks from the repo root:

```cmd
:: Check features parquet
python -c "import pandas as pd; df=pd.read_parquet('project/data/assembled_dataset/features_raw.parquet'); print('features_raw:', df.shape)"
:: Expected: features_raw: (420, 310)

:: Check VLM raw JSONs (should be 2100)
python -c "import pathlib; p=pathlib.Path('project/data/vlm_annotations/raw'); print('VLM JSONs:', len(list(p.glob('*.json'))))"
:: Expected: VLM JSONs: 2100

:: Check indicator vocabulary
python -c "import json; d=json.load(open('project/data/vlm_annotations/indicator_vocabulary.json')); print('Affordances in vocab:', [k for k in d if not k.startswith('_')])"
:: Expected: ['L059', 'L079', 'L091', 'L130', 'L141']

:: Check image manifest
python -c "import pandas as pd; df=pd.read_csv('project/configs/hypersim_image_manifest.csv'); print('Manifest rows:', len(df), '| Clusters:', df['cluster_assignment'].unique().tolist())"
:: Expected: Manifest rows: 420 | Clusters: ['kitchen', 'bedroom', ...]

:: Check images (spot-check one)
python -c "import pathlib; imgs=list(pathlib.Path('project/data/hypersim_pilot_420').rglob('*.jpg')); print('Images found:', len(imgs))"
:: Expected: Images found: 420
```

If any check fails, see [Section 8 — Troubleshooting](#8-troubleshooting).

---

## 6. Run the Pipeline

Run each script from the **repo root** in order. Each step must complete before the next.

> **Important:** Always activate the virtual environment first (`.venv\Scripts\activate.bat`).

---

### Step 1 — Assemble Dataset

**Script:** `project/src/models/assemble_dataset.py`
**Runtime:** ~2–5 minutes
**What it does:** Merges features + VLM scores + indicator-polarity binary features → 2,100-row `pilot_dataset.parquet`. Performs 70/10/20 scene-level split.

```cmd
python project\src\models\assemble_dataset.py
```

**Check the output:**

```cmd
:: Confirm the dataset was created
python -c "import pandas as pd; df=pd.read_parquet('project/data/assembled_dataset/pilot_dataset.parquet'); print('Dataset shape:', df.shape); print(df['split'].value_counts())"
:: Expected shape: (2100, ~500+)  split counts: train ~1470 / val ~210 / test ~420

:: Check the split summary
python -c "import json; s=json.load(open('project/data/assembled_dataset/split_summary.json')); print(json.dumps(s['splits'], indent=2))"
```

Files created:
```
project\data\assembled_dataset\pilot_dataset.parquet
project\data\assembled_dataset\split_summary.json
```

---

### Step 2 — Train LightGBM (Model B)

**Script:** `project/src/models/train_lgbm.py`
**Runtime:** 30–90 minutes (100 Optuna trials × 5 affordances × 5-fold CV)
**What it does:** Hyperparameter search + best-model retraining for each of the 5 affordances using raw structured features only.

```cmd
python project\src\models\train_lgbm.py
```

**Check the output:**

```cmd
:: Confirm models were saved
dir project\outputs\models\lgbm\

:: Print test results summary
python -c "import json; r=json.load(open('project/outputs/results/lgbm_training_summary.json')); [print(f\"{x['affordance_id']}: RMSE={x['test_metrics']['rmse']:.4f}\") for x in r]"
```

Files created per affordance (e.g. `L059\`):
```
project\outputs\models\lgbm\L059\lgbm_model.pkl
project\outputs\models\lgbm\L059\optuna_study.pkl
project\outputs\models\lgbm\L059\best_params.json
project\outputs\models\lgbm\L059\feature_importance.csv
project\outputs\results\lgbm_training_summary.json
```

---

### Step 3 — Train Indicator-Distilled LightGBM (Model D)

**Script:** `project/src/models/train_lgbm_indicators.py`
**Runtime:** 30–90 minutes
**What it does:** Same Optuna protocol as Model B but adds affordance-specific indicator-polarity binary features.

```cmd
python project\src\models\train_lgbm_indicators.py
```

**Check the output:**

```cmd
dir project\outputs\models\lgbm_indicators\

python -c "import json; r=json.load(open('project/outputs/results/lgbm_indicators_training_summary.json')); [print(f\"{x['affordance_id']}: RMSE={x['test_metrics']['rmse']:.4f}, indicators={x['n_indicator_features']}\") for x in r]"
```

Files created per affordance:
```
project\outputs\models\lgbm_indicators\L059\lgbm_indicators_model.pkl
project\outputs\models\lgbm_indicators\L059\optuna_study.pkl
project\outputs\models\lgbm_indicators\L059\best_params.json
project\outputs\models\lgbm_indicators\L059\feature_importance.csv
project\outputs\results\lgbm_indicators_training_summary.json
```

---

### Step 4 — Train CNN Baseline (Model A)

**Script:** `project/src/models/train_cnn.py`
**Runtime:** 2–4 hours with GPU / 12–24 hours CPU-only
**What it does:** 40 runs (5 affordances × ResNet-18/50 × 4 learning rates), saves best checkpoint per run.

> **Tip:** If runtime is a concern, the LightGBM experiments (Steps 2, 3, and the evaluation) are fully independent of the CNN. You can run Step 5 without waiting for the CNN — Experiment 1 will simply show NaN for the CNN column and print a warning.

```cmd
python project\src\models\train_cnn.py
```

**Monitor progress** — the script prints one line per run:

```
[1/40] L059 | resnet18 | LR=1e-03
  Epochs: 12/30 | Val RMSE: 1.4230 | Test RMSE: 1.5102 (147s)
```

**Check the output:**

```cmd
dir project\outputs\models\cnn\

python -c "import json; r=json.load(open('project/outputs/results/cnn_training_results.json')); print(f'Runs saved: {len(r)}')"
:: Expected: Runs saved: 40
```

Files created:
```
project\outputs\models\cnn\L059\L059_resnet18_lr1e-03_best.pt
... (40 .pt checkpoints total)
project\outputs\results\cnn_training_results.json
```

---

### Step 5 — Run All 7 Experiments

**Script:** `project/src/evaluation/run_experiments.py`
**Runtime:** 10–30 minutes (SHAP computation dominates)
**What it does:** Loads trained models, computes all metrics and figures, generates `summary.json`.

> Steps 2 and 3 must be complete. Step 4 (CNN) is optional — Experiment 1 and 3 will degrade gracefully with a warning if CNN results are missing.

```cmd
python project\src\evaluation\run_experiments.py
```

---

## 7. Check Outputs

After the full pipeline completes, verify outputs exist and are populated.

### Results CSVs

```cmd
dir project\outputs\results\
```

Expected files:

| File | Description |
|------|-------------|
| `pilot_dataset.parquet` *(in data/)* | Assembled ML dataset |
| `split_summary.json` *(in data/)* | Train/val/test counts |
| `lgbm_training_summary.json` | Model B per-affordance metrics |
| `lgbm_indicators_training_summary.json` | Model D per-affordance metrics |
| `cnn_training_results.json` | All 40 CNN run results |
| `experiment1_model_comparison.csv` | RMSE/MAE/r/ρ for all models |
| `experiment3_cnn_ablation.csv` | 2×4 CNN grid |
| `experiment4_feature_ablation.csv` | Progressive feature groups |
| `experiment5_indicator_value.csv` | Model B vs D + Wilcoxon p |
| `experiment6_shap_vs_vlm.csv` | SHAP vs VLM ranking Spearman ρ |
| `experiment7_segmentation_ablation.csv` | Placeholder (planned work) |
| `summary.json` | Master summary |

**Quick sanity check:**

```cmd
python -c "
import json
s = json.load(open('project/outputs/results/summary.json'))
print('Best overall model:', s['best_overall']['model'])
print('Mean RMSE:', s['best_overall']['mean_rmse'])
print('Best per affordance:')
for aff, info in s['best_model_per_affordance'].items():
    print(f'  {aff}: {info[\"model\"]} RMSE={info[\"rmse\"]}')
"
```

### Figures

```cmd
dir project\outputs\figures\
```

Expected figures (PNG + PDF pairs):

| Prefix | Experiment |
|--------|-----------|
| `experiment1_model_comparison.*` | Grouped bar chart |
| `experiment2_opt_history_{AFF}.*` | Optuna history per affordance |
| `experiment2_hp_importance_{AFF}.*` | HP importance per affordance |
| `experiment2_parallel_coords_{AFF}.*` | Parallel coordinates |
| `experiment3_cnn_heatmap_{AFF}.*` | CNN 2×4 RMSE heatmap |
| `experiment3_cnn_curves_{AFF}.*` | CNN training curves |
| `experiment4_feature_ablation.*` | Feature group line plot |
| `experiment5_indicator_value.*` | Paired B vs D bar chart |
| `experiment6_shap_bar_{AFF}.*` | SHAP bar plot |
| `experiment6_shap_beeswarm_{AFF}.*` | SHAP beeswarm plot |

Count total figures:

```cmd
:: PowerShell:
(Get-ChildItem project\outputs\figures\ -Filter *.png).Count
:: Expected: ~40+
```

---

## 8. Troubleshooting

### `ModuleNotFoundError` for any package

```cmd
:: Confirm the venv is active
where python
:: Must show .venv\Scripts\python.exe — if not, activate it:
.venv\Scripts\activate.bat

:: Then reinstall the missing package:
pip install <package-name>
```

### `FileNotFoundError: pilot_dataset.parquet`

You skipped Step 1 or it failed partway through.

```cmd
python project\src\models\assemble_dataset.py
```

Check for missing VLM JSONs in the output — it will print `WARNING: missing ...` for any gaps.

### `FileNotFoundError` for VLM JSON files

The `project\data\vlm_annotations\raw\` directory may not have transferred completely. Verify:

```cmd
python -c "import pathlib; print(len(list(pathlib.Path('project/data/vlm_annotations/raw').glob('*.json'))))"
:: Must be 2100
```

### LightGBM Optuna run is very slow

Reduce trials for a quick test. Edit `train_lgbm.py` line:

```python
N_TRIALS = 100   # change to 10 for a smoke test
```

### `torch.cuda.is_available()` returns `False` despite having an NVIDIA GPU

1. Confirm CUDA version matches PyTorch build: `nvidia-smi` → check "CUDA Version"
2. Reinstall PyTorch with the correct CUDA index URL (Step 4a)
3. Restart the terminal after driver/CUDA installation

### CNN training is too slow on CPU

Run only the LightGBM experiments and skip `train_cnn.py`. The evaluation script (`run_experiments.py`) handles missing CNN results gracefully — Experiments 1 and 3 will show NaN values for the CNN column.

Alternatively, set `MAX_EPOCHS = 5` and `PATIENCE = 2` in `train_cnn.py` as a smoke test.

### SHAP `TreeExplainer` error

Usually a numpy/shap version mismatch. Fix:

```cmd
pip install "numpy>=1.26,<2" "shap>=0.45.0" --force-reinstall
```

### `UnicodeDecodeError` when reading JSON files

The raw VLM JSONs are UTF-8. If you see decode errors, the files may have been transferred with encoding corruption. Add `encoding="utf-8"` to any `open()` calls, or check the file transfer method.

### PowerShell `python -c` quoting issues

PowerShell handles inner quotes differently from cmd. Prefer single-file scripts or use cmd for one-liner checks. Alternatively, save the check as a `.py` file and run it:

```cmd
python check.py
```

---

## Reference: File Locations

```
FINAL\
├── project\
│   ├── configs\
│   │   ├── affordance_definitions.json      # 5 affordance specs
│   │   ├── hypersim_image_manifest.csv      # 420 images + scene/cluster metadata
│   │   └── coco_to_taxonomy_map.json
│   ├── data\
│   │   ├── assembled_dataset\
│   │   │   ├── features_raw.parquet         # 420 × 310 input features
│   │   │   ├── feature_names.json           # column groupings
│   │   │   ├── pilot_dataset.parquet        # GENERATED by Step 1
│   │   │   └── split_summary.json           # GENERATED by Step 1
│   │   ├── hypersim_pilot_420\              # 420 scene images
│   │   ├── segmentation_outputs\            # Mask2Former outputs (420 JSON + NPZ)
│   │   └── vlm_annotations\
│   │       ├── raw\                         # 2100 VLM JSON files
│   │       └── indicator_vocabulary.json    # canonical indicator vocab
│   ├── src\
│   │   ├── models\
│   │   │   ├── assemble_dataset.py          # Step 1
│   │   │   ├── train_lgbm.py                # Step 2
│   │   │   ├── train_lgbm_indicators.py     # Step 3
│   │   │   └── train_cnn.py                 # Step 4
│   │   └── evaluation\
│   │       └── run_experiments.py           # Step 5
│   └── outputs\
│       ├── models\                          # GENERATED — model checkpoints
│       ├── figures\                         # GENERATED — PNG + PDF figures
│       └── results\                         # GENERATED — CSVs + JSONs
└── WINDOWS_SETUP_GUIDE.md                   # this file
```
