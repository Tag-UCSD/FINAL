"""Upload job orchestration helpers.

This module provides a minimal job abstraction on top of the existing
SciencePipeline so that large batches of images can be processed
asynchronously without holding an HTTP request open.

It intentionally avoids any task-queue dependency (Celery, RQ, etc.)
so that the same code can later be wired into a real worker
implementation without changing the public API.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional, Sequence, Tuple

from sqlalchemy.orm import Session

from backend.database.core import SessionLocal
from backend.models.jobs import UploadJob, UploadJobItem
from backend.models.assets import Image
from backend.science.pipeline import SciencePipeline, SciencePipelineConfig

logger = logging.getLogger(__name__)


UploadedRecord = Tuple[int, str, str]  # (image_id, storage_path, original_filename)


def create_upload_job_for_images(
    db: Session,
    user_id: Optional[int],
    records: Sequence[UploadedRecord],
) -> UploadJob:
    """Persist an UploadJob + UploadJobItems for the given images.

    Parameters
    ----------
    db:
        Open SQLAlchemy session.
    user_id:
        Optional admin user id who initiated the upload.
    records:
        Sequence of (image_id, storage_path, original_filename).
    """
    job = UploadJob(
        created_by_id=user_id,
        status="PENDING",
        total_items=len(records),
        completed_items=0,
        failed_items=0,
    )
    db.add(job)
    db.flush()

    for image_id, storage_path, original_name in records:
        item = UploadJobItem(
            job_id=job.id,
            image_id=image_id,
            filename=original_name,
            storage_path=storage_path,
            status="PENDING",
        )
        db.add(item)

    db.commit()
    db.refresh(job)
    logger.info("Created UploadJob %s with %d items", job.id, len(records))
    return job


def _run_upload_job_inner(session: Session, job_id: int) -> None:
    job = session.query(UploadJob).get(job_id)
    if not job:
        logger.warning("UploadJob %s not found; nothing to run.", job_id)
        return

    if not job.items:
        job.status = "COMPLETED"
        session.commit()
        logger.info("UploadJob %s has no items; marking as COMPLETED.", job_id)
        return

    job.status = "RUNNING"
    session.commit()

    # Science pipeline is instantiated once per job to reuse analyzers.
    config = SciencePipelineConfig(enable_all=True)
    pipeline = SciencePipeline(config=config, db=session)

    completed = 0
    failed = 0

    for item in list(job.items):
        if item.status not in ("PENDING", "RUNNING"):
            # Skip items that were already processed.
            continue

        item.status = "RUNNING"
        session.commit()

        try:
            ok = False
            if item.image_id is not None:
                ok = pipeline.process_image(item.image_id)
            else:
                logger.warning(
                    "UploadJobItem %s has no image_id; marking as FAILED.", item.id
                )

            if ok:
                item.status = "COMPLETED"
                completed += 1
            else:
                item.status = "FAILED"
                failed += 1

        except Exception as exc:  # pragma: no cover - defensive
            logger.exception(
                "Science pipeline failed for UploadJobItem %s: %s", item.id, exc
            )
            item.status = "FAILED"
            item.error_message = str(exc)
            failed += 1

        job.completed_items = completed
        job.failed_items = failed
        session.commit()

    if failed and not completed:
        job.status = "FAILED"
    elif failed:
        job.status = "COMPLETED_WITH_ERRORS"
    else:
        job.status = "COMPLETED"

    session.commit()
    logger.info(
        "UploadJob %s finished. completed=%d failed=%d",
        job.id,
        completed,
        failed,
    )


def run_upload_job(job_id: int) -> None:
    """Entry point for background workers.

    This function owns its own SessionLocal and is suitable for use with
    FastAPI's BackgroundTasks or an external worker process.
    """
    session = SessionLocal()
    try:
        _run_upload_job_inner(session, job_id)
    finally:
        session.close()
