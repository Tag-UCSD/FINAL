import logging
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

from backend.database.core import SessionLocal
from backend.database.core import get_db  # kept for FastAPI wiring
from backend.database.models import Image, Validation
from backend.science.core import AnalysisFrame
from backend.science.math.complexity import ComplexityAnalyzer
from backend.science.math.fractals import FractalAnalyzer
from backend.science.math.glcm import TextureAnalyzer
from backend.science.vision.materials import MaterialAnalyzer
from backend.science.math.color import ColorAnalyzer
from backend.science.vision import VisionProcessor
from backend.science.context.cognitive import CognitiveStateAnalyzer
from backend.science.context.social import SocialDispositionAnalyzer

logger = logging.getLogger(__name__)


def run_full_analysis(frame: AnalysisFrame, enable_expensive: bool = False) -> None:
    """Run the core science analyzers on an in-memory AnalysisFrame.

    This helper is designed for scripts and unit tests. It does *not* touch the
    database; it only populates `frame.attributes`.
    """

    # L0: fast numeric metrics
    ComplexityAnalyzer.analyze(frame)
    TextureAnalyzer.analyze(frame)
    try:
        d_score = FractalAnalyzer.fractal_dimension(frame.original_image)
        frame.add_attribute("fractal.D.global", float(d_score), confidence=0.8)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("FractalAnalyzer failed: %s", exc)

    # L1: material + color heuristics
    try:
        MaterialAnalyzer.analyze(frame)
    except Exception as exc:  # pragma: no cover
        logger.exception("MaterialAnalyzer failed: %s", exc)

    try:
        ColorAnalyzer.analyze(frame)
    except Exception as exc:  # pragma: no cover
        logger.exception("ColorAnalyzer failed: %s", exc)

    # L2/L3: higher-level context (kept light for now)
    try:
        CognitiveStateAnalyzer.analyze(frame)
    except Exception as exc:  # pragma: no cover
        logger.exception("CognitiveStateAnalyzer failed: %s", exc)

    try:
        SocialDispositionAnalyzer.analyze(frame)
    except Exception as exc:  # pragma: no cover
        logger.exception("SocialDispositionAnalyzer failed: %s", exc)


class SciencePipeline:
    """DB-backed science pipeline.

    Usage pattern (inside a FastAPI dependency or script):

    >>> db = SessionLocal()
    >>> pipeline = SciencePipeline(db)
    >>> pipeline.process_image(image_id=1)

    This will:
      1. Load the Image row.
      2. Load pixels from the data_store.
      3. Run the science analyzers.
      4. Persist attributes to Validation rows.
    """

    def __init__(self, db: Session):
        self.db = db

    async def process_image(self, image_id: int) -> bool:
        image: Optional[Image] = self.db.query(Image).filter(Image.id == image_id).one_or_none()
        if image is None:
            logger.warning("SciencePipeline: image %s not found", image_id)
            return False

        # Load pixels from the data_store
        try:
            from backend.services.data_store import load_image  # lazy import
            pil_image = load_image(image.storage_path)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("SciencePipeline: failed to load image %s: %s", image_id, exc)
            # Fallback: create a neutral gray image
            import PIL.Image as ImageLib
            pil_image = ImageLib.new("RGB", (256, 256), color=(128, 128, 128))

        frame = AnalysisFrame.from_pil(pil_image, image_id=image.id)

        # Run analyzers
        run_full_analysis(frame)

        # Persist attributes to Validation rows
        for attr in frame.attributes:
            validation = Validation(
                image_id=image.id,
                attribute_key=attr.key,
                value=float(attr.value),
                user_id=None,
                region_id=None,
                duration_ms=None,
                source="science_pipeline",
            )
            self.db.add(validation)

        self.db.commit()
        logger.info("SciencePipeline: processed image %s with %d attributes", image_id, len(frame.attributes))
        return True