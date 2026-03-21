
import logging
from typing import Dict, Any

import cv2
import numpy as np

from backend.science.core import AnalysisFrame
from backend.services.vlm import get_vlm_engine, StubEngine, get_cognitive_prompt, describe_vlm_configuration
from backend.services.costs import log_vlm_usage

logger = logging.getLogger(__name__)


class CognitiveStateAnalyzer:
    """Cognitive / affective analyzer driven by a Visual Language Model.

    The analyzer asks a VLM to rate the scene on:
    - Five core environmental-psychology dimensions (Kaplan & Kaplan + restoration)
    - Five affective / experiential dimensions (cozy, welcoming, tranquil, scary, jarring)
    """

    PROMPT = (
        "Analyze this architectural space as an environmental psychologist.\n\n"
        "Rate the following attributes from 0.0 (very low) to 1.0 (very high):\n\n"
        "1. coherence   – How organized and structured is the scene?\n"
        "2. complexity  – How much visual richness and variety is present?\n"
        "3. legibility  – How easy would it be to navigate and understand this space?\n"
        "4. mystery     – Does the environment promise more information if explored?\n"
        "5. restoration – Potential for stress recovery / mental restoration.\n\n"
        "Now also rate the affective tone of the space on these dimensions (0.0–1.0):\n"
        "6. cozy        – How cozy / snug / intimate does it feel?\n"
        "7. welcoming   – How welcoming / socially inviting does it feel?\n"
        "8. tranquil    – How calm / tranquil does it feel?\n"
        "9. scary       – How scary / threatening does it feel?\n"
        "10. jarring    – How visually or affectively jarring does it feel?\n\n"
        "Return ONLY valid JSON in the following form:\n"
        "{"
        "\"coherence\": float, "
        "\"complexity\": float, "
        "\"legibility\": float, "
        "\"mystery\": float, "
        "\"restoration\": float, "
        "\"cozy\": float, "
        "\"welcoming\": float, "
        "\"tranquil\": float, "
        "\"scary\": float, "
        "\"jarring\": float"
        "}\n"
    )

    def analyze(self, frame: AnalysisFrame) -> None:
        """Run VLM-based cognitive + affective analysis on the given frame.

        This is synchronous by design: the surrounding science pipeline is
        synchronous and this method may perform a blocking network call when
        a real VLM is configured, or return fast when using StubEngine.
        """
        try:
            if frame.original_image is None:
                logger.warning("CognitiveStateAnalyzer: frame has no original_image; skipping.")
                return

            # The pipeline already keeps images in RGB (np.ndarray).
            img = frame.original_image
            if not isinstance(img, np.ndarray):
                logger.warning("CognitiveStateAnalyzer: unsupported image type %r", type(img))
                return

            ok, buffer = cv2.imencode(".jpg", img)
            if not ok:
                logger.error("CognitiveStateAnalyzer: JPEG encoding failed; skipping.")
                return
            image_bytes = buffer.tobytes()

            engine = get_vlm_engine()
            prompt = get_cognitive_prompt(self.PROMPT)
            result: Dict[str, Any] = engine.analyze_image(image_bytes, prompt)

            # Stub / classroom path
            if isinstance(engine, StubEngine) or result.get("stub"):
                logger.info(
                    "CognitiveStateAnalyzer: running in STUB mode; "
                    "skipping cognitive/affect writes to avoid contaminating data."
                )
                # We deliberately do not write any cognitive.* or affect.* attributes here.
                # Downstream BN exports can treat the absence of these attributes as
                # "no VLM data available" instead of neutral 0.5 placeholders.
                return

            # Real data path: record cost for a single cognitive VLM call.
            try:
                cfg = describe_vlm_configuration()
                provider = cfg.get("provider", "auto")
                model_name = cfg.get("engine", type(engine).__name__)
                cost_per_1k = cfg.get("cost_per_1k_images_usd") or 0.0
                estimated_cost = float(cost_per_1k) / 1000.0 if cost_per_1k else 0.0
                log_vlm_usage(
                    provider=str(provider),
                    model_name=str(model_name),
                    cost_usd=estimated_cost,
                    meta={
                        "source": "science_pipeline_cognitive",
                        "image_id": getattr(frame, "image_id", None),
                        "cost_per_1k_images_usd": cost_per_1k,
                    },
                )
            except Exception:
                # Cost logging must never break the science pipeline.
                logger.debug("CognitiveStateAnalyzer: cost logging failed", exc_info=True)

            # Real data path
            def _clamp(x: float) -> float:
                try:
                    v = float(x)
                except Exception:
                    return 0.5
                if v < 0.0:
                    return 0.0
                if v > 1.0:
                    return 1.0
                return v

            # Core cognitive dimensions
            for key in ["coherence", "complexity", "legibility", "mystery", "restoration"]:
                if key in result:
                    frame.add_attribute(f"cognitive.{key}", _clamp(result[key]), confidence=0.9)

            # Affective tone dimensions
            for key in ["cozy", "welcoming", "tranquil", "scary", "jarring"]:
                if key in result:
                    frame.add_attribute(f"affect.{key}", _clamp(result[key]), confidence=0.9)

        except Exception as exc:
            logger.error("CognitiveStateAnalyzer failed: %s", exc)
            # We deliberately do not raise; the rest of the pipeline can still succeed.
            frame.add_attribute("cognitive.error", 1.0)