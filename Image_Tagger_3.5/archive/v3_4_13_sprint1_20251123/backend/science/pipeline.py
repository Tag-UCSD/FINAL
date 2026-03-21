"""
Science Pipeline Orchestrator v3.3 (Grand Jury Edition).
Integrates DepthAnalyzer and removed IsovistAnalyzer.
"""
import logging
from typing import Optional
import numpy as np
from sqlalchemy.orm import Session

try:
    import cv2
except ImportError:
    cv2 = None

from backend.models.assets import Image
from backend.models.annotation import Validation
from backend.science.core import AnalysisFrame

# v3.3 Modular Imports
from backend.science.math.color import ColorAnalyzer
from backend.science.math.complexity import ComplexityAnalyzer
from backend.science.math.glcm import TextureAnalyzer
from backend.science.math.fractals import FractalAnalyzer
from backend.science.math.symmetry import SymmetryAnalyzer
from backend.science.math.naturalness import NaturalnessAnalyzer
from backend.science.math.fluency import FluencyAnalyzer
from backend.science.spatial.depth import DepthAnalyzer # Replaces Isovist
from backend.science.context.cognitive import CognitiveStateAnalyzer

logger = logging.getLogger(__name__)

class SciencePipelineConfig:
    def __init__(self, enable_all: bool = True):
        self.enable_color = enable_all
        self.enable_complexity = enable_all
        self.enable_texture = enable_all
        self.enable_fractals = enable_all
        self.enable_spatial = enable_all
        self.enable_cognitive = False # Expensive, default off

class SciencePipeline:
    def __init__(self, db: Session, config: Optional[SciencePipelineConfig] = None):
        self.db = db
        self.config = config or SciencePipelineConfig()
        
        # Init Analyzers
        self.color = ColorAnalyzer()
        self.complexity = ComplexityAnalyzer()
        self.texture = TextureAnalyzer()
        self.fractals = FractalAnalyzer()
        self.symmetry = SymmetryAnalyzer()
        self.naturalness = NaturalnessAnalyzer()
        self.fluency = FluencyAnalyzer()
        self.spatial = DepthAnalyzer() # The new Spatial Engine
        self.cognitive = CognitiveStateAnalyzer()

    def process_image(self, image_id: int) -> bool:
        image_record = self.db.query(Image).get(image_id)
        if not image_record:
            logger.warning(f"Image {image_id} not found.")
            return False

        # Load Image
        rgb = self._load_image(image_record)
        if rgb is None: return False

        # Init Frame
        frame = AnalysisFrame(image_id=image_id, original_image=rgb)

        try:
            # L0: Physics & Basic Stats
            if self.config.enable_color: self.color.analyze(frame)
            if self.config.enable_complexity: self.complexity.analyze(frame)
            if self.config.enable_texture: self.texture.analyze(frame)
            if self.config.enable_fractals: self.fractals.analyze(frame)
            if self.config.enable_spatial: 
                self.symmetry.analyze(frame)
                self.naturalness.analyze(frame)
                self.spatial.analyze(frame) # Runs Depth/Clutter

            # L1: Perceptual (Dependent on L0)
            if self.config.enable_spatial:
                self.fluency.analyze(frame)

            # L2: Cognitive (VLM)
            if self.config.enable_cognitive:
                self.cognitive.analyze(frame)

        except Exception:
            logger.exception(f"Analysis failed for image {image_id}")
            return False

        self._save_results(image_id, frame.attributes)
        return True

    def _load_image(self, image_record: Image) -> Optional[np.ndarray]:
        # Simple local loader
        import os
        path = f"data_store/{image_record.storage_path}"
        if not os.path.exists(path): return None
        bgr = cv2.imread(path)
        if bgr is None: return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _save_results(self, image_id: int, attributes: dict) -> None:
        for key, value in attributes.items():
            if value != value: value = 0.0 # NaN check
            val = Validation(
                image_id=image_id,
                attribute_key=key,
                value=float(value),
                source="science_pipeline_v3.3"
            )
            self.db.add(val)
        self.db.commit()
