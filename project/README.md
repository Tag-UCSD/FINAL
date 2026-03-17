# Affordance Prediction from Indoor Scenes — Pilot Study

Knowledge distillation of affordance predictions from a Vision-Language Model (VLM) into lightweight tabular models, using 420 stratified Hypersim indoor scene images.

## Project Overview

**Phase 1 (Pilot):** 420 Hypersim images × 5 affordances, annotated via VLM (Qwen2-VL-7B-Instruct-AWQ), segmented via Mask2Former (COCO panoptic), features assembled into a Parquet dataset, and distilled into LightGBM and CNN models with SHAP explanations.

**5 Pilot Affordances:**
| ID | Name | Family | Primary Rooms |
|---|---|---|---|
| L059 | Sleep (Primary) | Rest & Recovery | Bedroom, Hotel Room |
| L079 | Cook (Daily) | Food & Drink | Kitchen, Commercial Kitchen |
| L091 | Computer Work (Solo) | Focused Knowledge Work | Home Office, Private/Open Office |
| L130 | Casual Conversation / Hangout | Leisure & Entertainment | Living Room, Hotel Lobby, Bar/Lounge |
| L141 | Yoga / Stretching | Movement & Fitness | Gym, Meditation Room, Living Room |

## Directory Structure

```
project/
├── configs/
│   ├── coco_to_taxonomy_map.json       # COCO 133 classes → taxonomy artefacts + affordance codes
│   ├── affordance_definitions.json     # Full definitions for 5 pilot affordances
│   └── hypersim_image_manifest.csv     # 420-image stratified selection manifest
├── data/
│   ├── hypersim_pilot_420/             # Downloaded Hypersim images (420 images on disk)
│   ├── segmentation_outputs/           # Mask2Former panoptic outputs (JSON + masks)
│   ├── vlm_annotations/                # Qwen2-VL affordance scores (JSON per image–affordance pair)
│   └── assembled_dataset/              # Final Parquet feature tables
├── scripts/
│   └── download_manifest_images.py     # Selective remotezip downloader (already run)
├── src/
│   ├── segmentation/                   # Mask2Former inference pipeline
│   ├── features/                       # Feature extraction from segmentation outputs
│   ├── vlm/                            # VLM annotation pipeline
│   │   ├── run_vlm_annotation.py       # Main annotation script (GPU required)
│   │   ├── extract_indicator_vocab.py  # Post-annotation indicator vocabulary builder
│   │   └── prompts/                    # System preamble, user template, few-shot examples
│   ├── models/                         # LightGBM + CNN training
│   └── evaluation/                     # SHAP analysis, metrics, figures
├── notebooks/
├── outputs/
│   ├── figures/
│   ├── models/
│   └── report/
├── requirements.txt
└── README.md
```

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Detectron2 (Mask2Former backend)

Detectron2 must be installed separately, matching your torch/CUDA version:

```bash
# Example for torch 2.3 + CUDA 11.8:
pip install 'git+https://github.com/facebookresearch/detectron2.git'
# Or check: https://detectron2.readthedocs.io/en/latest/tutorials/install.html
```

### 3. Images

The 420 images are already downloaded to `data/hypersim_pilot_420/`. To re-download or extend, run:

```bash
python scripts/download_manifest_images.py \
  --manifest configs/hypersim_image_manifest.csv \
  --output_dir data/hypersim_pilot_420 \
  --workers 12
```

This uses HTTP range requests via `remotezip` to extract only the specific JPEG frames from each remote scene zip (~100 MB total data transferred, vs ~700 GB for full zip downloads).

## Image Manifest — Stratification Strategy

The 420 images were selected from Hypersim metadata using the following strategy:

- **Source:** `metadata_images.csv` + `metadata_camera_trajectories.csv` from the official ml-hypersim repository
- **Filtering:** Only images with `included_in_public_release=True`; excluded trajectories labelled "OUTSIDE VIEWING AREA"
- **Camera preference:** `cam_00` tone-mapped final-quality images preferred
- **7 clusters × 60 images:** kitchen, bedroom, living_room, office, dining_room, bathroom, lobby_lounge
- **Scene diversity:** max 2 images per scene (3 for bathroom, which has only 27 scenes); 320 unique scenes represented
- **Sampling:** mid-trajectory frame selected per scene to capture representative viewpoints
- **Random seed:** 42 (fully reproducible)

| Cluster | Room Types Included | Count | Unique Scenes |
|---|---|---|---|
| kitchen | Kitchen | 60 | 40 |
| bedroom | Bedroom | 60 | 36 |
| living_room | Living Room | 60 | 60 |
| office | Office (all subtypes), Library, Lecture theater | 60 | 60 |
| dining_room | Dining Room, Restaurant | 60 | 46 |
| bathroom | Bathroom | 60 | 27 |
| lobby_lounge | Hotel Lobby, Hall, Hallway | 60 | 51 |

## Config Files

- **`coco_to_taxonomy_map.json`** — All 133 COCO panoptic classes under `classes` key, each mapped to a taxonomy artefact and relevant affordance codes. Reverse index in `_metadata.pilot_affordance_index`. Iterate via `data["classes"].items()`.
- **`affordance_definitions.json`** — Full definitions extracted from `Environment_Cognition_Taxonomy_Hierarchical_V2.7`. Includes definition, signature artefacts, spatial relations, positive/negative indicators, VLM scoring guidance, measurement items, and COCO proxy classes.

## Running the VLM Annotation (GPU required)

```bash
# Dry run to verify setup without GPU:
python src/vlm/run_vlm_annotation.py --dry-run

# Full annotation run (requires ≥16 GB VRAM):
python src/vlm/run_vlm_annotation.py

# Annotate specific affordances only:
python src/vlm/run_vlm_annotation.py --affordance-ids L079 L059

# Resume from checkpoint after interruption:
python src/vlm/run_vlm_annotation.py  # checkpoint is checked automatically
```

Outputs one JSON per image–affordance pair to `data/vlm_annotations/raw/`.

## Human Follow-Up Required

1. **Detectron2 install** — must be installed manually matching your torch/CUDA build (see Setup §2)
2. **VLM hardware** — Qwen2-VL-7B-Instruct-AWQ needs ≥16 GB VRAM; 420 × 5 = 2,100 calls ≈ 40–70 min on A100
3. **`src/segmentation/`** — Mask2Former inference pipeline not yet implemented; required before feature extraction and VLM prompt injection can use segmentation results
4. **Taxonomy indicator review** — `positive_indicators`/`negative_indicators` in `affordance_definitions.json` were inferred from `Key_Cues_2D` field text. Review against full taxonomy before finalising VLM prompts
