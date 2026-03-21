"""Smoketest for the science pipeline.

This test runs the SciencePipeline (or equivalent entrypoint) on a synthetic image
record and asserts that at least one science attribute is written to Validation.

It is marked as 'slow' by convention; teams can skip it in tight CI loops if needed.
"""

from pathlib import Path

import numpy as np
import cv2

import pytest

from backend.database.core import SessionLocal
from backend.models.assets import Image
from backend.models.annotation import Validation
from backend.science.pipeline import SciencePipeline


@pytest.mark.slow
def test_science_pipeline_writes_validation(tmp_path: Path):
    session = SessionLocal()
    try:
        # 1. Create a tiny synthetic image on disk.
        img_path = tmp_path / "science_smoke.png"
        arr = np.zeros((64, 64, 3), dtype=np.uint8)
        cv2.rectangle(arr, (8, 8), (56, 56), (255, 255, 255), thickness=2)
        cv2.imwrite(str(img_path), arr)

        # 2. Insert an Image row pointing to this file.
        image = Image(filename="science_smoke.png", storage_path=str(img_path))
        session.add(image)
        session.commit()
        session.refresh(image)
        image_id = image.id

        # 3. Run the science pipeline for this image.
        pipeline = SciencePipeline()
        ok = pipeline.run_for_image(image_id)
        assert ok is True

        # 4. Confirm that at least one science attribute exists in Validation.
        rows = (
            session.query(Validation)
            .filter(Validation.image_id == image_id)
            .filter(Validation.source.like("science_pipeline%"))
            .all()
        )
        assert rows, "Expected at least one science Validation row after running the pipeline"
    finally:
        session.close()
