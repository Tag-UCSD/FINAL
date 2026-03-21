# Image Tagger v3.4.12 - Enterprise Edition

This is the production-ready, micro-frontend architecture for the Image Tagger system.

## 🏗️ Architecture
* **Frontend:** Monorepo with 4 distinct React Apps (Workbench, Monitor, Admin, Explorer).
* **Backend:** Unified FastAPI service with PostgreSQL.
* **Infrastructure:** Docker Compose orchestration with Nginx Gateway.

## 🚀 Quick Start (The "Enterprise Go")

1.  **Ensure Docker is installed.**
2.  **Run:**
    ```bash
    cd deploy
    docker-compose up --build
    ```
3.  **Access the GUIs:**
    * **Research Explorer:** [http://localhost:8080/explorer](http://localhost:8080/explorer)
    * **Tagger Workbench:** [http://localhost:8080/workbench](http://localhost:8080/workbench)
    * **Supervisor Monitor:** [http://localhost:8080/monitor](http://localhost:8080/monitor)
    * **Admin Cockpit:** [http://localhost:8080/admin](http://localhost:8080/admin)

## 🧪 Running Tests

To verify the API logic without Docker:
1.  `pip install pytest httpx`
2.  `pytest tests/test_v3_api.py`

## 🤖 AI Collaboration Workflow

For guidelines on how to use LLMs (ChatGPT, Claude, Gemini, etc.) with this
repository — including ZIP + concatenated TXT expectations and Guardian
governance rules — see:

- `docs/AI_COLLAB_WORKFLOW.md`


## Quickstart & Seeding (v3.4.12)

From the repository root:

```bash
# Build containers, run seeds, and execute smoketests
./install.sh
```

The install script is intentionally small and opinionated. In the default configuration it will:

1. Build the containers (API, DB, frontend) using the `deploy/` Dockerfiles.
2. Run seeding scripts to populate core configuration:
   - `backend/scripts/seed_tool_configs.py`
   - `backend/scripts/seed_attributes.py`
3. Run cheap smoketests:
   - `scripts/smoke_api.py` (basic API shape)
   - `scripts/smoke_science.py` (science pipeline + composite indices and bins)

If any step fails, the script prints a clear message and returns a non-zero exit code.

Once `install.sh` completes successfully, you can point a browser at the frontend portal (typically `http://localhost:8000/index.html` or your configured frontend host) and choose the appropriate GUI:

- Tagger Workbench
- Supervisor Monitor
- Admin Cockpit
- Research Explorer

## Known Limitations (v3.4.12)

This v3 line is designed as a teachable, inspectable system rather than a fully productized SaaS. In particular:

- **Science modules** implement a deterministic, heuristic-based pipeline. Hooks for VLM / external models exist but are intentionally stubbed out by default to keep costs predictable.
- **Composite indices and bins** (for example, `science.visual_richness[_bin]`, `science.organized_complexity[_bin]`) are deliberately simple and intended as a starting point for BN and downstream modeling, not final scientific truth.
- **CI workflow** is included as a template. It shows how to wire Guardian and basic tests but may require adaptation to your specific infrastructure (Python versions, DB configuration, secrets).
- **Empty dashboards** in Monitor / Explorer generally mean you have not yet:
  - Run the seeding scripts, and
  - Processed any images through the science pipeline, or
  - Collected enough validation data for IRR / Tag Inspector to be informative.

## CI Skeleton

The repository includes a minimal GitHub Actions workflow under `.github/workflows/ci_v3.yml`. It is meant as a starting point for teams who want to:

1. Guard against repository drift via `scripts/guardian.py verify`.
2. Exercise core API contracts via `pytest tests/test_v3_api.py`.
3. Optionally, run `scripts/smoke_science.py` as a lightweight end-to-end science check.

The CI recipe assumes a standard Python environment and may need adjustments to match your Docker / DB setup or your preferred dependency-management strategy.

## Minimal test suite

Before cutting a new release or making substantial changes, run:

- `pytest tests/test_v3_api.py` – API and RBAC sanity checks.
- `pytest tests/test_guardian.py` – governance and Guardian behaviour.
- `pytest tests/test_bn_export_smoke.py` – BN export tied to Validation.
- `pytest tests/test_workbench_smoke.py` – Tagger Workbench basic flow.
- `pytest tests/test_explorer_smoke.py` – Explorer attributes and search.

For deeper science verification, you can also run `pytest -m slow` to include
`tests/test_science_pipeline_smoke.py`, which exercises the full science
pipeline on a synthetic image.

## Science Debug Layers

For an explanation of the edge-map and overlay debug views in the Explorer,
see `docs/SCIENCE_DEBUG_LAYERS.md`. This document is intended as a teaching
aid to help students understand how Canny edge detection and parameter
settings relate to visual complexity and the science pipeline.

## Production Deployment

For guidance on running Image Tagger beyond a single development laptop
(e.g. on a lab server or departmental host), see
`docs/PRODUCTION_DEPLOYMENT.md`. It covers environment variables,
volumes, HTTPS / reverse proxy structure, and a minimal security
checklist.



## Optional: AutoInstaller + AI Copilot (v1.3.0)

You can also use the shared **AutoInstaller + AI Copilot** kit:

```bash
bash infra/turnkey_installer_v1.3/installer/install.sh
```

This will:
- Write a detailed log to `logs/install.log`
- Run the native `install.sh` for Image Tagger via `infra/turnkey_installer_v1.3/installer_config.json`
- Optionally allow an AI copilot to propose remediation steps from the logs:

```bash
python ai/installer_copilot.py --logfile logs/install.log --out logs/ai_plan.json --dry-run 1 --provider none
```
