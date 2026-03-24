# COGS 185 Final Project

**Distilling VLM-Derived Indoor Affordance Scores from Synthetic Scene Images**

Taggert Smith | Department of Cognitive Science, UC San Diego

This repository contains two integrated systems:

1. **`project/`** -- A research pipeline that trains lightweight models to predict how suitable indoor scenes are for specific activities (affordances), using knowledge distilled from a vision-language model.
2. **`Image_Tagger_3.5/`** -- A full-stack computer vision application for analyzing architectural interior images, now extended with the affordance prediction pipeline from the research experiments.

---

## Repository Layout

```
FINAL/
├── README.md                  ← you are here
├── project/                   ← research experiments & affordance pipeline
│   ├── configs/               ← affordance definitions, COCO class maps, image manifest
│   ├── data/                  ← Hypersim images, segmentation outputs, VLM annotations
│   ├── src/                   ← all pipeline source code
│   │   ├── segmentation/      ← Mask2Former panoptic segmentation
│   │   ├── features/          ← 310-dim feature extraction from segmentation masks
│   │   ├── vlm/               ← Qwen2-VL annotation pipeline + prompts
│   │   ├── models/            ← LightGBM & CNN training scripts
│   │   └── evaluation/        ← 7-experiment evaluation suite
│   ├── outputs/
│   │   ├── models/            ← trained model checkpoints (.pkl)
│   │   ├── results/           ← experiment CSVs (model comparison, ablation, etc.)
│   │   ├── figures/           ← publication-ready plots
│   │   └── report/            ← LaTeX paper (main.tex → main.pdf)
│   └── scripts/               ← data download utilities
│
├── Image_Tagger_3.5/          ← full-stack image analysis application
│   ├── backend/               ← FastAPI + PostgreSQL backend
│   │   ├── science/           ← computer vision pipeline (20+ analyzers)
│   │   ├── api/               ← REST API endpoints
│   │   ├── models/            ← database models (SQLAlchemy)
│   │   ├── services/          ← VLM, auth, storage services
│   │   └── scripts/           ← seeding & training utilities
│   ├── frontend/              ← React monorepo (4 apps)
│   ├── deploy/                ← Docker Compose + Dockerfiles + nginx
│   ├── docs/                  ← deployment & usage guides
│   └── install.sh             ← one-command setup
│
├── planning docs/             ← project plans & taxonomy spreadsheet
└── ml-hypersim-main/          ← Hypersim dataset utilities (reference)
```

---

## Research Experiments (`project/`)

### What it does

The research pipeline answers: *Can a lightweight tabular model predict how suitable an indoor scene is for a given activity, using only object-level features from panoptic segmentation?*

Five affordances are studied:

| Code | Activity | Family |
|------|----------|--------|
| L059 | Sleep (Primary) | Rest & Recovery |
| L079 | Cook (Daily) | Food & Drink |
| L091 | Computer Work (Solo) | Focused Knowledge Work |
| L130 | Casual Conversation | Leisure & Entertainment |
| L141 | Yoga / Stretching | Movement & Fitness |

### Pipeline stages

1. **Image acquisition** -- 420 images from the Hypersim synthetic indoor dataset, stratified across 7 room clusters.
2. **Panoptic segmentation** -- Mask2Former (COCO-133 classes) extracts object masks, centroids, and areas.
3. **VLM annotation** -- Qwen2-VL-7B scores each image-affordance pair on a 1-7 Likert scale and outputs structured semantic indicators.
4. **Feature engineering** -- 310 raw features (object presence/counts, pairwise distances, room aggregates) plus 1,248 binary indicator features distilled from VLM annotations.
5. **Model training** -- LightGBM regressors optimized with Optuna (100 trials, 5-fold CV).

### Key results

| Model | Features | Mean RMSE | Best affordance |
|-------|----------|-----------|-----------------|
| CNN (ResNet-18) | Raw pixels | ~1.13 | -- |
| LightGBM (Model B) | 310 raw | ~0.97 | L079 (Cook) |
| **Indicator-LGBM (Model D)** | **1,558 raw + indicators** | **0.76** | **L079 (RMSE 0.52, r=0.92)** |

Model D is statistically significantly better than Model B for most affordances (Wilcoxon p < 0.05).

### Experiment output files

All results are in `project/outputs/results/`:

| File | Contents |
|------|----------|
| `experiment1_model_comparison.csv` | Head-to-head: CNN vs LightGBM vs Indicator-LGBM |
| `statistical_tests.csv` | Wilcoxon p-values and Cohen's d effect sizes |
| `metric_confidence_intervals.csv` | 95% CIs for RMSE, MAE, Pearson r |
| `indicator_permutation_test.csv` | Proof that indicator features add value beyond raw features |
| `classification_f1_results.csv` | Binary classification at threshold = 4.0 |
| `vlm_score_diagnostics.csv` | VLM score distributions by affordance |

The paper is at `project/outputs/report/main.pdf`.

---

## Image Tagger 3.5 (`Image_Tagger_3.5/`)

Image Tagger is a Docker-based web application for analyzing indoor architectural images. It runs a layered computer vision pipeline and provides a browser UI for exploring results.

### Prerequisites

- **Docker** and **Docker Compose** (v2+)
- At least 8 GB RAM available for Docker (the OneFormer segmentation model is large)
- Optionally, one or more VLM API keys for higher-level analysis:
  - `GEMINI_API_KEY` (Google Gemini Flash -- recommended, cheapest)
  - `OPENAI_API_KEY` (GPT-4o)
  - `ANTHROPIC_API_KEY` (Claude)

### Quick Start

```bash
cd Image_Tagger_3.5
bash install.sh
```

This single command will:
1. Check that Docker is installed.
2. Run structural integrity checks (Guardian governance scripts).
3. Build and start three Docker containers (PostgreSQL, FastAPI backend, React + Nginx frontend).
4. Seed the database with the attribute taxonomy and VLM model configurations.
5. Run smoke tests to verify everything is working.

Once complete, open your browser to:

| URL | What it is |
|-----|------------|
| **http://localhost:8080/explorer/** | Research Explorer -- browse images, view science metrics, debug overlays |
| http://localhost:8080/workbench/ | Tagger Workbench -- annotate images with human judgments |
| http://localhost:8080/admin/ | Admin Cockpit -- manage VLM models, monitor costs |
| http://localhost:8080/monitor/ | Supervisor Monitor -- track annotation progress & inter-rater reliability |
| http://localhost:8080/api/docs | Swagger API documentation (interactive) |

The default entry point (`http://localhost:8080/`) redirects to the Explorer.

### Architecture Overview

```
Browser (:8080)
  │
  └─ Nginx reverse proxy
       ├─ /explorer/   → React Explorer SPA
       ├─ /workbench/  → React Workbench SPA
       ├─ /admin/      → React Admin SPA
       ├─ /monitor/    → React Monitor SPA
       └─ /api/        → FastAPI backend (:8000)
                             │
                             ├─ Science Pipeline (20+ analyzers)
                             │   ├─ L0: Color, texture, complexity, fractals
                             │   ├─ L1: Depth, spatial frequency, fluency
                             │   ├─ L1.5: OneFormer segmentation
                             │   ├─ L1.8: Affordance prediction (NEW)
                             │   └─ L2: Cognitive/affective VLM analysis
                             │
                             └─ PostgreSQL 15
```

### The Science Pipeline

The backend runs a layered analysis pipeline on each image. Lower layers are fast heuristics; higher layers use deep learning or VLM calls.

| Layer | Analyzers | What they compute |
|-------|-----------|-------------------|
| L0 | Color, Complexity, Texture, Fractals, Symmetry | Perceptual statistics (CIELAB color, Shannon entropy, GLCM, fractal dimension) |
| L1 | Depth, Naturalness, Fluency, Spatial Frequency | Monocular depth estimation, visual processing fluency, FFT analysis |
| L1.5 | Segmentation (OneFormer) | Semantic + panoptic instance masks with 150 ADE20K classes |
| L1.8 | **Affordance Prediction** | Activity suitability scores for 5 affordances (Sleep, Cook, Work, Conversation, Yoga) |
| L2 | Cognitive, Semantic Tags, Architectural Patterns | VLM-based environmental psychology dimensions, design styles, architectural features |

Each analyzer writes normalized attributes to the frame, which are then persisted to the database and visible in the Explorer UI.

### Enabling Affordance Prediction

The affordance analyzer is **opt-in**. It requires the OneFormer segmentation layer (L1.5) to run first.
The Explorer UI also computes and caches affordance scores on demand for image cards and the image detail view.

**From the API / debug UI:**

Visit `http://localhost:8080/api/v1/debug/images/{image_id}/affordance` to get a JSON response with affordance scores for any image in the database.
The Explorer also exposes `GET /api/v1/explorer/images/{image_id}/affordance` for the GUI.

**From Python code:**

```python
from backend.science.pipeline import SciencePipeline, SciencePipelineConfig

config = SciencePipelineConfig()
config.enable_segmentation = True   # required -- runs OneFormer
config.enable_affordance = True     # enables affordance prediction

pipeline = SciencePipeline(db=session, config=config)
pipeline.process_image(image_id)

# Scores are saved to the Validation table as:
#   affordance.L059       (1.0-7.0 Likert scale)
#   affordance.L059_norm  (0.0-1.0 normalized)
#   ... etc for L079, L091, L130, L141
```

**Standalone (no database):**

```python
from backend.science.context.affordance import predict_affordances_from_image
import cv2

image = cv2.cvtColor(cv2.imread("my_room.jpg"), cv2.COLOR_BGR2RGB)
scores = predict_affordances_from_image(image)
# {'L059': 4.73, 'L079': 0.95, 'L091': 2.51, 'L130': 3.11, 'L141': 2.36}
```

### Retraining the Affordance Models

The pre-trained LightGBM models are bundled in `backend/science/data/affordance_models/`. To retrain from the pilot dataset (e.g., after updating training data):

```bash
cd Image_Tagger_3.5
pip install lightgbm pandas optuna   # if not already installed
python -m backend.scripts.train_affordance_models
```

This reads `project/data/assembled_dataset/pilot_dataset.parquet`, trains one LightGBM regressor per affordance with Optuna hyperparameter optimization (50 trials, 5-fold CV), and saves the models.

### User Roles & Authentication

The app uses a simple header-based role system suitable for classroom use:

| Role | Access | Auth |
|------|--------|------|
| `tagger` | Workbench (annotate images) | Default -- no auth needed |
| `scientist` | Explorer (read-only browsing) | Header: `X-User-Role: scientist` |
| `supervisor` | Monitor (annotation oversight) | Header: `X-Auth-Token: <API_SECRET>` |
| `admin` | Admin cockpit (model/cost management) | Header: `X-Auth-Token: <API_SECRET>` |

The default `API_SECRET` is `dev_secret_key_change_me`. For any non-local deployment, set a real secret via the `SECRET_KEY` environment variable in `deploy/docker-compose.yml`.

### VLM Configuration

L2 analyzers (cognitive, semantic, architectural) require a VLM backend. The app auto-detects available providers from environment variables. Set one or more in `deploy/docker-compose.yml`:

```yaml
environment:
  - GEMINI_API_KEY=your_key_here      # cheapest option
  - OPENAI_API_KEY=your_key_here      # GPT-4o
  - ANTHROPIC_API_KEY=your_key_here   # Claude
```

A built-in cost tracker enforces a hard budget limit (default $15 USD, configurable via `VLM_HARD_LIMIT_USD`) to prevent runaway API spending.

The affordance analyzer (L1.8) now prefers the best-performing project model: indicator-augmented LightGBM (Model D). That path uses OneFormer segmentation plus runtime VLM indicator extraction when a VLM API key is configured. If no VLM is configured, the app falls back to the bundled raw-feature LightGBM model.

### Stopping & Restarting

```bash
# Stop all containers (data is preserved in Docker volumes)
cd Image_Tagger_3.5/deploy
docker compose down

# Restart
docker compose up -d

# Full reset (deletes database and image data)
docker compose down -v
```

### Development (without Docker)

For local development of the backend:

```bash
cd Image_Tagger_3.5

# Start PostgreSQL separately (or use a local instance)
# Set DATABASE_URL in your environment

# Install Python dependencies
pip install fastapi uvicorn sqlalchemy psycopg2-binary torch transformers \
            lightgbm pandas scikit-image scipy supervision

# Run the backend
uvicorn backend.main:app --reload --port 8000

# In another terminal, run the frontend dev servers
cd frontend
npm install
npm run dev:all    # starts all 4 apps on ports 3001-3004
```

### Useful API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| GET | `/api/v1/debug/images/{id}/affordance` | Affordance scores for an image |
| GET | `/api/v1/debug/images/{id}/room` | Room type detection overlay |
| GET | `/api/v1/debug/images/{id}/edges` | Canny edge detection overlay |
| GET | `/api/v1/debug/images/{id}/segmentation` | OneFormer segmentation overlay |
| GET | `/api/v1/debug/images/{id}/materials` | Gemini material detection |
| POST | `/api/v1/explorer/search` | Search images |
| GET | `/api/v1/features/` | Browse feature ontology |
| GET | `/docs` | Interactive Swagger UI |

---

## Other Directories

| Directory | Contents |
|-----------|----------|
| `planning docs/` | Project plans (.docx) and the V2.7 Activity Affordances taxonomy spreadsheet (.xlsx) |
| `ml-hypersim-main/` | Reference copy of the Hypersim dataset download utilities |
