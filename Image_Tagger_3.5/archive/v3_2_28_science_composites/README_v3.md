# Image Tagger v3.2.27 (Enterprise) – Multi-GUI HITL Tagging System

This repository hosts the **Image Tagger v3** backend + frontends used for
architecture / CNfA-style tagging workflows. It is designed to support:

- Human-in-the-loop (**HITL**) annotation for architectural images.
- A minimal but **truthful science pipeline** that computes basic visual metrics.
- Monitoring and governance for classroom and research deployments.

The system is split into **four GUIs**, all backed by a single FastAPI backend.

---

## 1. System Overview

### 1.1 Four GUIs and their roles

1. **Tagger Workbench** (worker GUI)
   - Target user: student / junior researcher taggers.
   - Purpose: fast, keyboard-driven binary tagging (one attribute question at a time).
   - Backend endpoints:
     - `/v1/annotation/*`

2. **Supervisor Monitor** (monitor GUI)
   - Target user: supervisors / TAs / lab managers.
   - Purpose: monitor team velocity and disagreement patterns.
   - Backend endpoints:
     - `/v1/monitor/velocity`
     - `/v1/monitor/irr`
     - `/v1/monitor/image/{image_id}/validations`

3. **Admin Cockpit** (admin GUI)
   - Target user: PI / course staff / system admins.
   - Purpose: configure tools, budgets, and export training data.
   - Backend endpoints:
     - `/v1/admin/models`
     - `/v1/admin/budget`
     - `/v1/admin/training/export`

4. **Research Explorer** (explorer GUI)
   - Target user: researchers / advanced students.
   - Purpose: search images by attributes and export training datasets.
   - Backend endpoints:
     - `/v1/explorer/attributes`
     - `/v1/explorer/search`
     - `/v1/explorer/export`

Each GUI is served as a separate SPA-style frontend under `frontend/apps/`.

---

## 2. Install and Run

### 2.1 One-command install

The recommended entry point is:

```bash
./install.sh
```

From a machine with **Docker** and **docker-compose** installed, this script will:

1. Build and start the Docker services defined in `deploy/docker-compose.yml`.
2. Run seed scripts inside the API container:
   - `backend/scripts/seed_tool_configs.py`
   - `backend/scripts/seed_attributes.py`
3. Run smoketests:
   - `python -m scripts.smoke_api` (API health and core routers).
   - `python -m scripts.smoke_science` (science pipeline + DB write).
4. Run a small pytest suite:
   - `pytest tests/test_v3_api.py`
   - `pytest tests/test_guardian.py` (optional in CI).

If any of these steps fail, `install.sh` will exit non-zero and print a
human-readable error message indicating which stage failed.

> **Note**: The install script will **not** silently ignore errors; it is
> deliberately opinionated so students notice when something is wrong.

### 2.2 Guardian integration

This repo ships with a light-weight governance layer:

- Config: `v3_governance.yml`
- Tool: `scripts/guardian.py`
- Baseline file: `governance.lock`

Typical usage:

```bash
# First time on a new clone (create baseline)
python3 scripts/guardian.py freeze

# Subsequent checks (ensure no drift)
python3 scripts/guardian.py verify
```

- `freeze()` creates or refreshes the baseline `governance.lock`.
- `verify()` checks:
  - All **critical files** exist and exceed a minimum size.
  - All **protected files** still exist and have unchanged hashes.
  - No unexpected root-level files appear when `prevent_new_root_files` is enabled.

If `governance.lock` does not yet exist, `verify()` returns success and logs
that there is no baseline. This avoids stranding first-time users.

---

## 3. Science Pipeline: “Minimum Viable Truthful”

The **science pipeline** in `backend/science/` is designed to be:

- Deterministic (no random outputs).
- Fast enough for classroom / batch use.
- Scientifically modest but honest.

### 3.1 What it does today

The pipeline operates on each `Image` and records attributes into the DB via
`Validation` rows (source = `"science_pipeline_v3.3"`). Current capabilities:

- **Color metrics** (`backend/science/math/color.py`)
  - Luminance / brightness summary.
  - Simple warm–cool index (channel balance).
  - Basic saturation measures.

- **Texture metrics** (`backend/science/math/glcm.py`)
  - GLCM-based contrast, homogeneity, and related texture stats.

- **Complexity / structure metrics** (`backend/science/math/complexity.py`)
  - Edge density and simple entropy-like measures.

- **Fractals** (`backend/science/math/fractals.py`)
  - Approximate fractal dimension, tuned for architectural images.

- **Materials heuristics** (`backend/science/vision/materials.py`)
  - HSV-rule heuristics for wood, vegetation, and related materials.

- **Context and perception** (`backend/science/context/*.py`, `backend/science/perception.py`)
  - Light-weight, rule-based proxies for social density and cognitive load.
  - These are explicitly labeled as *heuristics* in code.

The orchestrator is `backend/science/pipeline.py`:

- Loads image pixels via `Image.storage_path`.
- Creates an `AnalysisFrame`.
- Runs the analyzers above.
- Writes derived attributes as `Validation` rows.

The smoketest `scripts/smoke_science.py` verifies this end-to-end.

### 3.2 What is *not* promised

To keep the documentation honest:

- The current science stack **does not** claim:
  - Full VLM-based semantic understanding.
  - Production-grade depth maps or 3D reasoning.
- API stubs for more advanced science are present, but calls are either:
  - Guarded behind config flags, or
  - Implemented as clearly-marked deterministic heuristics.

Any future VLM / depth integration should:

- Be implemented behind explicit configuration.
- Update both the code and this README before being advertised in GUIs.

---

## 4. Monitor, Admin, Explorer – Data Truthfulness

### 4.1 Supervisor Monitor

Monitor endpoints:

- `/v1/monitor/velocity`
  - Aggregates `Validation` rows per user over a recent time window.
  - Reports images validated, average dwell time (`duration_ms`), and inferred status.

- `/v1/monitor/irr`
  - Computes simple inter-rater reliability (IRR) based on overlapping validations.
  - Uses a conservative “agreement ratio” between raters (exact value match).

- `/v1/monitor/image/{image_id}/validations`
  - Returns per-user validations for a specific image to drive the Tag Inspector.

These views are **live**: they read from the same `validations` table that
Tagger Workbench and the science pipeline use. If no data exists, views may be
empty but they are *not* backed by hard-coded demo arrays.

### 4.2 Admin & Explorer

- Admin Cockpit:
  - Lists and updates `ToolConfig` rows.
  - Exports training data using `backend/services/training_export.py`.

- Research Explorer:
  - Lists registered attributes (keys + descriptions).
  - Uses real DB-backed image search where possible.
  - Exports JSONL / JSON training sets via the training exporter.

---

## 5. CI: Example GitHub Actions Workflow

For teams using GitHub, this repo includes an example CI workflow at:

- `.github/workflows/ci.yml`

It performs the following checks on push / pull request:

1. Set up Python.
2. Install backend dependencies (via `requirements.txt`, if present).
3. Run Guardian verification:
   - `python scripts/guardian.py verify`
4. Run API and Guardian tests:
   - `pytest -q tests/test_v3_api.py tests/test_guardian.py`

You may adapt or expand this workflow to add linting, type-checking, or
end-to-end front-end tests.

---

## 6. Known Limitations (Truth-in-Advertising)

To avoid over-claiming:

- Science metrics are **heuristic and approximate**, suitable for:
  - Exploratory CNfA research.
  - Classroom projects and demos.
- VLM, depth, and other heavy models are:
  - Either not enabled by default, or
  - Present as stubs that should be clearly marked before use.
- The current CI workflow focuses on:
  - Governance (Guardian) and
  - Core API wiring tests.

If you extend science, RBAC, or front-end behavior, please:

1. Update `v3_governance.yml` if new critical paths are added.
2. Extend tests in `tests/` and, if appropriate, the CI workflow.
3. Refresh this README so students and collaborators have an accurate picture
   of what the system actually does.
