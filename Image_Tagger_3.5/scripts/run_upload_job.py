#!/usr/bin/env python
"""Run a single upload job's science pipeline.

Usage:
    python -m scripts.run_upload_job <job_id>
    or
    python scripts/run_upload_job.py <job_id>

This script is a thin wrapper around `backend.services.upload_jobs.run_upload_job`.
It is intended for use by a worker process in more advanced deployments.
"""

import argparse
import logging

from backend.database import SessionLocal  # ensures DB is configured
from backend.services.upload_jobs import run_upload_job


logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run an upload job's science pipeline")
    parser.add_argument("job_id", type=int, help="ID of the UploadJob to run")
    args = parser.parse_args()

    job_id = args.job_id
    logger.info("Running upload job %s", job_id)
    # `run_upload_job` manages its own SessionLocal internally.
    run_upload_job(job_id)
    logger.info("Completed upload job %s", job_id)


if __name__ == "__main__":
    main()
