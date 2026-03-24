# Image Tagger Explorer

A web application for exploring interior architecture images and inspecting features extracted by computer vision pipelines. Built with FastAPI (Python) and React, deployed via Docker.

## Prerequisites

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) installed and running

Optional (for VLM-powered features like material detection and semantic tagging):

- A Gemini API key (`GEMINI_API_KEY` environment variable)

## Quick Start

```bash
# From this directory:
bash install.sh
```

This builds three Docker containers (PostgreSQL, FastAPI backend, Nginx + React frontend), seeds the attribute taxonomy, and starts the system.

Once running, open **http://localhost:8080/explorer/** in your browser.

Note: in some environments, **http://localhost:8080/** may not redirect correctly. If the root URL fails, go directly to **http://localhost:8080/explorer/**.

## Using the Explorer

The Explorer is the single interface for browsing and analyzing images.

### Gallery View

- **Search**: Type a query in the search bar and press Enter, or click Search.
- **Filters**: Toggle the sidebar to filter by attribute categories (color, texture, spatial, etc.).
- **Debug Modes**: Cycle through visualization overlays on the gallery thumbnails — Edges, Overlay, Depth, Complexity, Segmentation, Room, Materials, Materials2. Each mode renders the image through a different computer vision analyzer in real time.
- **Affordance Ratings**: Each image card can show 1-7 ratings for Sleep, Cook, Work, Conversation, and Yoga as affordance scores are computed and cached.
- **Export Cart**: Check images to add them to your cart, then click Export Dataset to download a JSON file of their validation records.

### Single-Image Detail View

Click any image to open the full detail modal:

- **Navigation**: Arrow keys or Prev/Next buttons to move between images.
- **Debug Modes**: Press keys 1-8 to switch visualization modes. Sliders for edge thresholds, overlay opacity, and segmentation confidence appear when relevant.
- **Affordance Ratings**: Right sidebar shows the five 1-7 affordance predictions for the current image.
- **Science Attributes**: Bottom-left panel shows all computed feature values grouped by analyzer (color, texture, fractal, fluency, segmentation, etc.), each with a mini bar chart.
- **Human Validations**: Bottom-right panel shows any manual annotation records.
- **Tags**: Right sidebar shows tags with provenance tooltips (imported vs. pipeline-derived, with confidence scores).
- **Cart**: Press C to toggle the current image in/out of the export cart.
- **Close**: Press Escape or click the X button.

### Seeding Sample Data

If the database is empty on first load, Explorer shows a "Seed Sample Dataset" button. Click it to import ~9,500 sample image URLs from the bundled dataset. This populates the gallery with browsable images.

## Architecture

```
Image_Tagger_3.5/
  backend/                  # FastAPI + SQLAlchemy (Python)
    api/                    # Route handlers
      v1_discovery.py       #   Explorer: search, export, attributes, image detail, seeding
      v1_debug.py           #   Visualization: edges, depth, complexity, segmentation, room, materials
      v1_features.py        #   Feature registry browser
    science/                # Computer vision pipeline
      math/                 #   Color, complexity, texture, fractals, symmetry, naturalness, fluency
      vision/               #   OneFormer segmentation, material detection, room classification
      semantics/            #   VLM-based style and cognitive tagging
      context/              #   Kaplan cognitive dimensions, affordance prediction, social space
      spatial/              #   Monocular depth, isovist visibility analysis
      pipeline.py           #   Orchestrator that runs all analyzers on an image
    models/                 # SQLAlchemy ORM (Image, Validation, Attribute, User, Region)
    database/               # DB engine setup + seed data (google_images_import.json)
    scripts/                # seed_attributes.py, import_images_from_json.py
  frontend/                 # React + Vite + Tailwind
    src/
      App.jsx               # Explorer gallery with search, filters, debug modes, cart
      ImageDetailModal.jsx  # Full image inspector with science attributes + human validations
      lib/                  # Shared components (ApiClient, Header, Button, Toast, MaintenanceOverlay)
  deploy/                   # Docker Compose + Dockerfiles + Nginx config
  install.sh                # One-command setup
```

## API Reference

The backend serves interactive docs at **http://localhost:8080/api/docs** when running.

Key endpoints:

| Endpoint | Method | Description |
|---|---|---|
| `/v1/explorer/search` | POST | Search images with filters and pagination |
| `/v1/explorer/attributes` | GET | List the attribute taxonomy for filters |
| `/v1/explorer/images/{id}/detail` | GET | Full image detail (science attrs + human validations + tags) |
| `/v1/explorer/export` | POST | Export validation records for selected image IDs |
| `/v1/explorer/seed` | POST | Seed database with sample images |
| `/v1/debug/images/{id}/edges` | GET | Canny edge map PNG (params: t1, t2) |
| `/v1/debug/images/{id}/depth` | GET | Monocular depth map PNG |
| `/v1/debug/images/{id}/complexity` | GET | Edge density heatmap PNG |
| `/v1/debug/images/{id}/segmentation` | GET | OneFormer semantic segmentation overlay PNG |
| `/v1/debug/images/{id}/room` | GET | Places365 room type classification overlay PNG |
| `/v1/debug/images/{id}/materials` | GET | Gemini VLM material detection overlay PNG |
| `/v1/debug/images/{id}/materials2` | GET | OneFormer + SigLIP2 material identification overlay PNG |
| `/v1/debug/images/{id}/affordance` | GET | Affordance prediction scores (JSON) |
| `/v1/features/` | GET | Browse the feature/attribute ontology |

## Science Pipeline

The pipeline (`backend/science/pipeline.py`) runs a configurable set of analyzers on each image:

| Analyzer | Module | Features Produced |
|---|---|---|
| Color | `math/color.py` | CIELAB color space metrics |
| Complexity | `math/complexity.py` | Canny edge density |
| Texture | `math/glcm.py` | Gray-Level Co-occurrence Matrix |
| Fractals | `math/fractals.py` | Fractal dimension estimate |
| Symmetry | `math/symmetry.py` | Bilateral symmetry score |
| Naturalness | `math/naturalness.py` | Nature-likeness heuristics |
| Fluency | `math/fluency.py` | Visual entropy, clutter density |
| Depth | `spatial/depth.py` | MonoDepth2 monocular depth |
| Segmentation | `vision/segmentation.py` | OneFormer semantic + panoptic |
| Room Detection | `vision/room_detection.py` | Places365 room type classification |
| Materials (VLM) | `vision/materials.py` | Gemini Flash material/finish identification |
| Materials (CLIP) | `vision/clip_material.py` | OneFormer + SigLIP2 material identification |
| Semantic Tags | `semantics/semantic_tags_vlm.py` | VLM-derived style and cognitive tags |
| Cognitive | `context/cognitive.py` | Kaplan complexity, mystery, legibility, coherence |
| Affordance | `context/affordance.py` | Activity suitability scores (5 pre-trained LightGBM models) |

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `GEMINI_API_KEY` | Enables Gemini Flash material detection and VLM semantic tagging | (none - falls back to stub) |
| `GOOGLE_API_KEY` | Alternative to GEMINI_API_KEY | (none) |
| `DATABASE_URL` | PostgreSQL connection string | Set by docker-compose |
| `IMAGE_STORAGE_ROOT` | Local image file storage directory | `data_store` |

## Stopping and Restarting

```bash
# Stop all containers
cd deploy && docker-compose down

# Restart (preserves database)
cd deploy && docker-compose up -d

# Full reset (destroys database)
cd deploy && docker-compose down -v && docker-compose up -d --build
```

## Local Development (without Docker)

```bash
# Backend
pip install fastapi uvicorn sqlalchemy psycopg2-binary pydantic pillow numpy opencv-python-headless scipy scikit-image requests PyYAML torch torchvision transformers timm supervision lightgbm pandas
export DATABASE_URL=postgresql://tagger:tagger_pass@localhost:5432/image_tagger_v3
uvicorn backend.main:app --reload --port 8000

# Frontend
cd frontend && npm install && npm run dev
```

The frontend dev server proxies `/api` requests to `localhost:8000`.
