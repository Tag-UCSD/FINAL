# Upload Jobs & Science Orchestrator

This document describes how the Image Tagger upload job system works in v3.4.51 and how to monitor or run jobs.

## Overview

- When an admin uploads images via the Admin Cockpit, the `/api/v1/admin/upload` endpoint:
  - Persists the images to disk.
  - Creates `Image` rows in the database.
  - Creates an `UploadJob` row plus one `UploadJobItem` per image.
  - Enqueues a background job to run the science pipeline for each image in the batch.

- The science pipeline call uses `SciencePipeline(config=SciencePipelineConfig(enable_all=True), db=session)` and:
  - Computes math-based metrics (fractal, texture, color, complexity).
  - Runs the cognitive VLM analyzer and architectural-patterns VLM analyzer as configured.
  - Logs VLM cost usage to the `ToolUsage` ledger.

## Monitoring in the Admin Cockpit

The Admin Cockpit now exposes an **Upload jobs** panel:

- Shows the most recent jobs with:
  - Job id
  - Status (`PENDING`, `RUNNING`, `COMPLETED`, `COMPLETED_WITH_ERRORS`, `FAILED`)
  - Progress (completed / total items)
  - Error count and short summary
- The panel has a **Refresh** button to pull the latest state from `/api/v1/admin/upload/jobs`.
- After a bulk upload completes, the panel will automatically refresh via the `handleUploadCompleted` callback.

This gives TAs and admins visibility into whether large batches are still running, have failed, or are fully processed.

## API Endpoints

- `GET /api/v1/admin/upload/jobs?limit=20`
  - Returns a list of the most recent jobs and high-level status counters.
- `GET /api/v1/admin/upload/jobs/{job_id}`
  - Returns detailed information for a single job, including per-item statuses.

Both endpoints require an admin role (`X-User-Role: admin`).

## Running Jobs from a Worker Process

The primary execution path is FastAPI's in-process `BackgroundTasks`. For more advanced deployments, you can run jobs from a separate worker using:

```bash
python -m scripts.run_upload_job <job_id>
```

This script:

- Creates its own database session.
- Calls `backend.services.upload_jobs.run_upload_job(job_id)`.
- Updates job and item statuses in-place.

In a real multi-process deployment, you would typically:

- Point a task queue (e.g. Celery, RQ, or a simple cron) at `scripts.run_upload_job`.
- Dispatch a job id from the API layer into the queue.
- Let the worker process handle the science pipeline execution.

For classroom use, the built-in `BackgroundTasks` runner is usually sufficient.
