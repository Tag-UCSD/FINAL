"""Science pipeline orchestrator for Image Tagger v3.3."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from sqlalchemy.orm import Session

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

from backend.models.assets import Image
from backend.models.annotation import Validation
from backend.science.core import AnalysisFrame
from backend.science.math.color import ColorAnalyzer
from backend.science.math.complexity import ComplexityAnalyzer
from backend.science.math.glcm import TextureAnalyzer
from backend.science.math.fractals import FractalAnalyzer
from backend.science.spatial.depth import DepthAnalyzer
from backend.science.context.cognitive import CognitiveStateAnalyzer

logger = logging.getLogger(__name__)


class SciencePipelineConfig:
    """Runtime configuration flags for the science pipeline."""

    def __init__(
        self,
        enable_color: bool = True,
        enable_complexity: bool = True,
        enable_texture: bool = True,
        enable_fractals: bool = True,
        enable_spatial: bool = True,
        enable_cognitive: bool = False,
        image_root: str = "data_store",
    ) -> None:
        self.enable_color = enable_color
        self.enable_complexity = enable_complexity
        self.enable_texture = enable_texture
        self.enable_fractals = enable_fractals
        self.enable_spatial = enable_spatial
        self.enable_cognitive = enable_cognitive
        self.image_root = image_root


class SciencePipeline:
    """Orchestrates science analyzers over Image records."""

    def __init__(self, db: Session, config: Optional[SciencePipelineConfig] = None):
        self.db = db
        self.config = config or SciencePipelineConfig()
        self.color = ColorAnalyzer()
        self.complexity = ComplexityAnalyzer()
        self.texture = TextureAnalyzer()
        self.fractals = FractalAnalyzer()
        self.spatial = DepthAnalyzer()
        self.cognitive = CognitiveStateAnalyzer()

    def process_image(self, image_id: int) -> bool:
        """Run science analyses for a single image_id.

        Returns True on success, False on any fatal error. Partial attribute
        extraction is allowed; we commit whatever was computed.
        """
        image_record = self.db.query(Image).get(image_id)
        if image_record is None:
            logger.warning("SciencePipeline: image %s not found", image_id)
            return False

        rgb = self._load_image(image_record)
        if rgb is None:
            logger.warning("SciencePipeline: could not load pixels for %s", image_id)
            return False

        frame = AnalysisFrame(image_id=image_id, original_image=rgb)

        try:
            if self.config.enable_color:
                self.color.analyze(frame)
            if self.config.enable_complexity:
                self.complexity.analyze(frame)
            if self.config.enable_texture:
                self.texture.analyze(frame)
            if self.config.enable_fractals:
                self.fractals.analyze(frame)
            if self.config.enable_spatial:
                self.spatial.analyze(frame)
            if self.config.enable_cognitive:
                self.cognitive.analyze(frame)
        except Exception:
            logger.exception("SciencePipeline: error while analyzing image %s", image_id)

        self._save_results(image_id, frame.attributes)
        return True

    def _load_image(self, image_record: Image) -> Optional[np.ndarray]:
        """Load an RGB uint8 image for the given record.

        This implementation assumes a local file under config.image_root.
        In future we can replace this with a storage abstraction.
        """
        rel = Path(image_record.storage_path)
        path = Path(self.config.image_root) / rel
        if not path.exists():
            logger.error("SciencePipeline: file does not exist: %s", path)
            return None
        if cv2 is None:
            logger.error("SciencePipeline: cv2 not available, cannot load image")
            return None

        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def _save_results(self, image_id: int, attributes: dict) -> None:
        for key, value in attributes.items():
            val = Validation(
                image_id=image_id,
                attribute_key=key,
                value=value,
                source="science_pipeline_v3.3",
            )
            self.db.add(val)
        self.db.commit()