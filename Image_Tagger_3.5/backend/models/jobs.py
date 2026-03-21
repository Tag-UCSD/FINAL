from __future__ import annotations

from typing import List, Optional

from sqlalchemy import String, Integer, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.core import Base
from backend.database.mixins import TimestampMixin


class UploadJob(Base, TimestampMixin):
    """Represents a batch upload + science run.

    This is intentionally lightweight: it tracks only aggregate progress and
    a short error summary. Per-image details live in UploadJobItem.
    """

    __tablename__ = "upload_jobs"

    id: Mapped[int] = mapped_column(primary_key=True)

    # Optional link back to the admin user who initiated the job.
    created_by_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("users.id"), nullable=True
    )

    status: Mapped[str] = mapped_column(String(32), default="PENDING", index=True)
    total_items: Mapped[int] = mapped_column(Integer, default=0)
    completed_items: Mapped[int] = mapped_column(Integer, default=0)
    failed_items: Mapped[int] = mapped_column(Integer, default=0)

    error_summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    items: Mapped[List["UploadJobItem"]] = relationship(
        "UploadJobItem", back_populates="job", cascade="all, delete-orphan"
    )


class UploadJobItem(Base, TimestampMixin):
    """Individual item within an UploadJob.

    Each item corresponds to a single Image row and tracks the status of
    science-pipeline processing for that image.
    """

    __tablename__ = "upload_job_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    job_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("upload_jobs.id"), index=True
    )

    image_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("images.id"), nullable=True
    )

    filename: Mapped[str] = mapped_column(String(255))
    storage_path: Mapped[str] = mapped_column(String(512))

    status: Mapped[str] = mapped_column(String(32), default="PENDING", index=True)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    job: Mapped["UploadJob"] = relationship("UploadJob", back_populates="items")
