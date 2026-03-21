"""DB-backed smoke test for the science pipeline and composite indices.

This script verifies that:

  * The database is reachable.
  * At least one Image record exists.
  * At least one science run completes without crashing.
  * Core composite attributes and their bins are present in Validation.

It is intentionally conservative and cheap to run; it is not a full benchmark.
"""

import sys
from pathlib import Path

from sqlalchemy.orm import Session

from backend.database.core import SessionLocal
from backend.models.assets import Image
from backend.models.annotation import Validation
from backend.science.pipeline import SciencePipeline


REQUIRED_KEYS = [
    "science.visual_richness",
    "science.organized_complexity",
    "science.visual_richness_bin",
    "science.organized_complexity_bin",
]


def _choose_image(session: Session) -> Image:
    image = session.query(Image).first()
    if image is None:
        raise SystemExit("[smoke_science] No Image rows found; seed at least one test image.")
    return image


def _run_pipeline(session: Session, image: Image) -> None:
    pipeline = SciencePipeline(session=session)
    # Prefer process_image, with fallback to run_for_image for legacy APIs.
    if hasattr(pipeline, "process_image"):
        ok = pipeline.process_image(image.id)
    else:
        ok = pipeline.run_for_image(image.id)  # type: ignore[attr-defined]
    if not ok:
        raise SystemExit(f"[smoke_science] Science pipeline reported failure for image_id={image.id}")


def _assert_composites_present(session: Session, image: Image) -> None:
    rows = (
        session.query(Validation)
        .filter(Validation.image_id == image.id)
        .filter(Validation.attribute_key.in_(REQUIRED_KEYS))
        .all()
    )
    present = {row.attribute_key for row in rows}
    missing = [k for k in REQUIRED_KEYS if k not in present]
    if missing:
        raise SystemExit(
            "[smoke_science] Missing expected science attributes for image_id=%s: %s"
            % (image.id, ", ".join(missing))
        )
    print("[smoke_science] Composite indices OK for image_id=%s: %s" % (image.id, ", ".join(sorted(present))))


def main() -> int:
    session = SessionLocal()
    try:
        image = _choose_image(session)
        _run_pipeline(session, image)
        _assert_composites_present(session, image)
        print("[smoke_science] SUCCESS")
        return 0
    finally:
        try:
            session.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
