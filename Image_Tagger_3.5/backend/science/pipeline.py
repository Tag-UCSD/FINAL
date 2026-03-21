"""
Science Pipeline Orchestrator v3.4 (OneFormer Edition).
Integrates DepthAnalyzer and updated SegmentationAnalyzer using OneFormer.
"""
import logging
from typing import Optional
import numpy as np
from sqlalchemy.orm import Session

from backend.database.core import SessionLocal

try:
    import cv2
except ImportError:
    cv2 = None

from backend.models.assets import Image
from backend.models.annotation import Validation
from backend.science.core import AnalysisFrame

# v3.4 Modular Imports
from backend.science.math.color import ColorAnalyzer
from backend.science.math.complexity import ComplexityAnalyzer
from backend.science.math.glcm import TextureAnalyzer
from backend.science.math.fractals import FractalAnalyzer
from backend.science.math.symmetry import SymmetryAnalyzer
from backend.science.math.naturalness import NaturalnessAnalyzer
from backend.science.math.fluency import FluencyAnalyzer
from backend.science.spatial.depth import DepthAnalyzer  # Replaces Isovist
from backend.science.context.cognitive import CognitiveStateAnalyzer
from backend.science.semantics.semantic_tags_vlm import SemanticTagAnalyzer
from backend.science.vision.segmentation import SegmentationAnalyzer  # Now uses OneFormer
from backend.science.vision.materials import GeminiMaterialAnalyzer
from backend.science.context.affordance import AffordanceAnalyzer

logger = logging.getLogger(__name__)


class SciencePipelineConfig:
    def __init__(self, enable_all: bool = True):
        self.enable_color = enable_all
        self.enable_complexity = enable_all
        self.enable_texture = enable_all
        self.enable_fractals = enable_all
        self.enable_spatial = enable_all
        # Expensive L2 analyzers are explicit opt-ins.
        self.enable_cognitive = False  # Cognitive VLM (Kaplan-style dimensions)
        self.enable_semantic = False   # Semantic VLM (style.*, room_function.*)
        # OneFormer segmentation — opt-in (runs full semantic+panoptic merge pipeline)
        self.enable_segmentation = False
        self.segmentation_use_semantic = True   # include semantic coverage metrics
        self.segmentation_use_panoptic = True   # include instance metrics + merged masks
        # Gemini Flash material detection (VLM) — opt-in
        self.enable_materials_vlm = False
        # OneFormer + SigLIP2 material identification — opt-in
        # Requires enable_segmentation=True (reuses cached detections from frame.metadata)
        self.enable_clip_materials = False
        # Affordance prediction (environmental activity suitability) — opt-in
        # Requires enable_segmentation=True (reuses cached detections from frame.metadata)
        self.enable_affordance = False

class SciencePipeline:
    def __init__(
        self, 
        db: Optional[Session] = None, 
        session: Optional[Session] = None, 
        config: Optional[SciencePipelineConfig] = None
    ):
        self.db = db or session or SessionLocal()
        self._owns_session = (db is None and session is None)
        self.config = config or SciencePipelineConfig()
        
        # Init Analyzers
        self.color = ColorAnalyzer()
        self.complexity = ComplexityAnalyzer()
        self.texture = TextureAnalyzer()
        self.fractals = FractalAnalyzer()
        self.symmetry = SymmetryAnalyzer()
        self.naturalness = NaturalnessAnalyzer()
        self.fluency = FluencyAnalyzer()
        self.spatial = DepthAnalyzer()  # The new Spatial Engine
        self.cognitive = CognitiveStateAnalyzer()
        self.semantic = SemanticTagAnalyzer()
        self.segmentation = SegmentationAnalyzer()  # OneFormer semantic+panoptic
        self.materials_vlm = GeminiMaterialAnalyzer()
        self.affordance = AffordanceAnalyzer()
        # clip_material is instantiated lazily on first use (heavy model load)
        self._clip_material_pipeline = None

    def process_image(self, image_id: int) -> bool:
        image_record = self.db.query(Image).get(image_id)
        if not image_record:
            logger.warning(f"Image {image_id} not found.")
            return False

        # Load Image
        rgb = self._load_image(image_record)
        if rgb is None:
            return False

        # Init Frame
        frame = AnalysisFrame(image_id=image_id, original_image=rgb)

        try:
            # L0: Physics & Basic Stats
            if self.config.enable_color:
                self.color.analyze(frame)
            if self.config.enable_complexity:
                self.complexity.analyze(frame)
            if self.config.enable_texture:
                self.texture.analyze(frame)
            if self.config.enable_fractals:
                self.fractals.analyze(frame)
            if self.config.enable_spatial:
                self.symmetry.analyze(frame)
                self.naturalness.analyze(frame)
                self.spatial.analyze(frame)  # Runs Depth/Clutter

            # L1: Perceptual (Dependent on L0)
            if self.config.enable_spatial:
                self.fluency.analyze(frame)

            # L1.5: Vision (OneFormer semantic + panoptic segmentation)
            if self.config.enable_segmentation:
                self.segmentation.analyze(
                    frame,
                    use_semantic=self.config.segmentation_use_semantic,
                    use_panoptic=self.config.segmentation_use_panoptic,
                )

            # L1.6: Materials — Gemini Flash VLM
            if self.config.enable_materials_vlm:
                self.materials_vlm.analyze(frame)

            # L1.7: Materials — OneFormer + SigLIP2 (clip_material pipeline)
            # Reuses cached OneFormer detections from L1.5; skips re-inference.
            if self.config.enable_clip_materials:
                if not self.config.enable_segmentation:
                    logger.warning(
                        "enable_clip_materials=True but enable_segmentation=False; "
                        "running segmentation first."
                    )
                    self.segmentation.analyze(frame, use_semantic=True, use_panoptic=True)
                from backend.science.vision.clip_material import MaterialIdentificationPipeline
                if self._clip_material_pipeline is None:
                    self._clip_material_pipeline = MaterialIdentificationPipeline.from_frame_models(frame)
                    if self._clip_material_pipeline is None:
                        self._clip_material_pipeline = MaterialIdentificationPipeline.from_pretrained()
                mat_results = self._clip_material_pipeline.run_from_frame(
                    frame, show_voting_report=False
                )
                # Store top material score per instance as pipeline attributes
                for r in mat_results:
                    safe = r["class_name"].replace(" ", "_")
                    frame.add_attribute(
                        f"material.{safe}_{r['instance_idx']}_top", r["top_score"]
                    )

            # L1.8: Affordance prediction (requires L1.5 segmentation)
            if self.config.enable_affordance:
                if not self.config.enable_segmentation:
                    logger.warning(
                        "enable_affordance=True but enable_segmentation=False; "
                        "running segmentation first."
                    )
                    self.segmentation.analyze(frame, use_semantic=True, use_panoptic=True)
                self.affordance.analyze(frame)

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
        if not os.path.exists(path):
            return None
        bgr = cv2.imread(path)
        if bgr is None:
            return None
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    def _save_results(self, image_id: int, attributes: dict) -> None:
        for key, value in attributes.items():
            if value != value:  # NaN check
                value = 0.0
            val = Validation(
                image_id=image_id,
                attribute_key=key,
                value=float(value),
                source="science_pipeline_v3.4"
            )
            self.db.add(val)
        self.db.commit()