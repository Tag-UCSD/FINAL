"""DB-backed smoke test for the science pipeline.

This script verifies that:

  * The database is reachable.
  * At least one Image record exists.
  * SciencePipeline.process_image(image_id) runs without crashing.
  * At least one Validation row with source="science_pipeline_v3.3"
    is written for that image.

Exit codes:
  0 = success
  1 = failure (no images, DB error, or no attributes written)

Typical usage inside the API container:

  python -m scripts.smoke_science
"""

from __future__ import annotations

import sys
from typing import Optional

from sqlalchemy.orm import Session

from backend.database.session import SessionLocal
from backend.models.assets import Image
from backend.models.annotation import Validation
from backend.science.pipeline import SciencePipeline, SciencePipelineConfig


def _pick_image(db: Session) -> Optional[Image]:
    """Pick an image for the smoketest.

    Preference order:
    1. An image that has *no* science_pipeline_v3.3 validations yet,
       so we exercise a fresh run.
    2. Otherwise, the first Image in the table.
    """
    img = (
        db.query(Image)
        .outerjoin(
            Validation,
            (Validation.image_id == Image.id)
            & (Validation.source == "science_pipeline_v3.3"),
        )
        .filter(Validation.id.is_(None))
        .order_by(Image.id)
        .first()
    )
    if img is not None:
        return img

    return db.query(Image).order_by(Image.id).first()


def main() -> int:
    try:
        db = SessionLocal()
    except Exception as exc:
        print(f"[smoke_science] FAILED: could not create DB session: {exc}")
        return 1

    try:
        image = _pick_image(db)
        if image is None:
            print(
                "[smoke_science] FAILED: no Image rows found in database. "
                "Seed at least one image before running this smoke test."
            )
            return 1

        image_id = image.id
        print(f"[smoke_science] Using image_id={image_id}")

        cfg = SciencePipelineConfig()
        pipeline = SciencePipeline(db=db, config=cfg)

        ok = pipeline.process_image(image_id=image_id)
        if not ok:
            print(
                "[smoke_science] FAILED: SciencePipeline.process_image "
                "returned False"
            )
            return 1

        count = (
            db.query(Validation)
            .filter(
                Validation.image_id == image_id,
                Validation.source == "science_pipeline_v3.3",
            )
            .count()
        )
        if count <= 0:
            print(
                "[smoke_science] FAILED: no Validation rows written for "
                f"image_id={image_id} with source='science_pipeline_v3.3'"
            )
            return 1

        print(
            f"[smoke_science] SUCCESS: {count} validation rows written "
            f"for image_id={image_id}"
        )
        return 0
    finally:
        try:
            db.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())