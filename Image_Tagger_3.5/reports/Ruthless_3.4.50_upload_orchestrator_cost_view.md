# Ruthless Audit — Image Tagger v3.4.50 (Upload Orchestrator + Cost View)

Target build: **Image_Tagger_v3.4.50_upload_orchestrator_cost_view_full.zip**  
Scope: backend, admin frontend, governance, science pipeline as exercised by the new upload job orchestrator and cost ledger.

## 1. Executive Verdict (GO / NO-GO)

**Verdict: GO (with high-priority workflow warnings)**

*The system is buildable, installable, and scientifically usable. The new upload orchestrator and cost ledger are wired correctly and do not introduce rot. However, the upload job system is "dark" (no dedicated Admin UI panel yet), and background execution is limited to FastAPI's in-process `BackgroundTasks`.*

- **Installability:** `install.sh` + governance guards still enforce hollow-repo, syntax, critical-import, and canon checks.
- **Science:** Image-level math metrics and cognitive VLM analysis are still functional and now log costs; arch-pattern VLM also logs costs.
- **New features:** Uploads now create `UploadJob` + `UploadJobItem` rows and enqueue a background science run; cost ledger exposes a daily time-series; Admin shows a cost history chart.
- **Workflow gap:** There is no first-class UI to monitor upload jobs; admins must call `/admin/upload/jobs` manually.

## 2. Kill List (Blockers)

No **hard** technical blockers for classroom / lab usage were found. The following are **soft blockers** for large-scale, multi-user deployments:

- **K1 — Orchestrator observability gap (soft blocker):**  
  - There is no dedicated Admin panel to monitor upload jobs, even though the backend exposes `/admin/upload/jobs` and `/admin/upload/jobs/{job_id}`.
  - Risk: Admins cannot easily see whether a big batch is still running, has failed, or is completed without manual API calls.
  - Impact: For large cohorts, this undermines trust in the orchestrator and makes triage slow.

- **K2 — BackgroundTasks as the only worker (soft blocker):**  
  - `run_upload_job(job_id)` is only wired from FastAPI `BackgroundTasks` in `upload_images`.
  - If the app process is restarted after upload but before the background task completes, the job may remain `RUNNING` or `PENDING` without a retry mechanism.
  - Impact: For long-running, large jobs this is fragile; for classroom-scale jobs it is acceptable.

These do **not** prevent GO for lab and teaching deployments but should be addressed before "industrial" multi-process deployment.

## 3. Architecture & Code Health

**Status: Solid, no rot**

- **Job models:** `UploadJob` and `UploadJobItem` are canonical SQLAlchemy models with timestamps, status fields, and FK links to `users` and `images`. No ellipses, no `# STUB:` markers.
- **Service layer:** `backend/services/upload_jobs.py` cleanly encapsulates:
  - Job creation (`create_upload_job_for_images`).
  - A reusable `_run_upload_job_inner(session, job_id)` that can be reused by real workers.
  - A `run_upload_job(job_id)` entry point holding its own `SessionLocal`.
- **Science pipeline:** Orchestrator uses `SciencePipeline(config=SciencePipelineConfig(enable_all=True), db=session)` and `process_image(image_id)`; no short-circuits, no half-wired calls.
- **Cost ledger:** Both `CognitiveStateAnalyzer` and `ArchPatternsVLMAnalyzer` now call `describe_vlm_configuration()` and `log_vlm_usage(...)` in their real-data paths; stub paths still early-return.
- **No rot:** `compileall` passes on `backend/`; no stray `...` or uncontrolled stubs in critical modules.

## 4. Admin Frontend (UX / Workflow)

**Status: Good foundations, missing one key panel**

- **Budget & cost:**
  - Existing Budget panel shows `total_spent`, `hard_limit`, and kill switch status.
  - New `CostHistoryCard` renders a minimal daily bar chart, last-7-days cost, and delta vs previous 7 days.
  - Failure mode: When there is no usage, card shows a clear informational message instead of blank UI.
- **Upload workflow:**
  - The upload form still works as before but now receives a `job_id` from `AdminUploadResult`.
  - There is not yet a dedicated "Upload jobs" table or progress indicator tied to this `job_id`.

**UX Warnings:**

- **W1 — Hidden job id:** The response’s `job_id` is not surfaced explicitly in the Admin UI as a clickable entry point to a job monitor.
- **W2 — No "Recent upload jobs" panel:** Admins have no dashboard view of job statuses, progress, or error counts.

## 5. Science & Metrics

**Status: Strong math + cognitive, semantic still stubbed**

- **Math layers (L0/L1):** Fractal, texture, color, and complexity analyzers remain fully implemented and stable.
- **Cognitive VLM:** Cognitive state analyzer uses the configured VLM, logs costs, and writes attributes + evidence strings into the frame and DB.
- **Architectural patterns VLM:**
  - VLM-backed architectural pattern analyzer is live, emitting attributes like `arch.pattern.prospect_strong`, with confidence values and evidence.
  - Costs are logged per call into the ledger.
- **Semantic tags (L3/L4):**
  - Style and room-function semantics are still in the registry but marked as stubs; they are not yet wired to real VLM classification.

**Science warning:**

- **S1 — Semantic registry still vaporware:** The new orchestrator does not change the fact that high-level semantic tags (`style.*`, `room_function.*`) remain unimplemented. For pure CNfA reasoning this is acceptable, but for “full pipeline semantics” it is a known TODO, not a regression.

## 6. Governance & Rot Detection

**Status: Excellent**

- `v3_governance.yml`, `governance.lock`, and `release.keep.yml` are intact and unchanged by this sprint.
- `install.sh` still runs the 5-guard gauntlet (hollow repo, program integrity, syntax, critical imports, canon).
- New modules (`backend/models/jobs.py`, `backend/services/upload_jobs.py`) contain no stubs and respect the existing patterns.

## 7. Recommended Next Sprint (from this audit)

Prioritised based on this audit:

1. **Upload Job Monitor UI (high priority):**
   - Add an Admin "Upload jobs" panel:
     - Table: job id, status, total/completed/failed, created time, error summary.
     - Link from the upload success message: “Job #123 created — view in Upload Jobs.”
     - Auto-refresh or a manual “Refresh jobs” button.

2. **Optional worker hook (medium priority):**
   - Add a script or CLI entry point to run `run_upload_job(job_id)` from a separate worker process.
   - Document how to wire this into a real queue in `docs/UPLOAD_JOBS_README.md`.

3. **Semantic tag activation (larger science sprint):**
   - Wire one or two high-value semantic tag families (e.g., `style.modern`, `room_function.living_room`) to a real VLM classifier and feed their costs into the ledger.

This sprint (v3.4.50) is therefore **GO** for classroom and lab use, with the above steps recommended for the next iterations.
