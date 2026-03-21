"""
VLM-based semantic tagging for interior style and room function.

This analyzer turns a single Visual Language Model (VLM) call into a small
set of semantic attributes:

- style.* (modern, traditional, minimalist, scandinavian, industrial, rustic,
  bohemian, farmhouse, japandi)
- spatial.room_function.* (living_room, kitchen, bedroom, home_office, bathroom)

Design goals:
- Mirror the CognitiveStateAnalyzer pattern (JPEG encoding, stub vs. real path).
- Log cost for each real VLM call via backend.services.costs.log_vlm_usage.
- Keep the science pipeline resilient: failures record metadata but do not
  raise exceptions.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

import cv2
import numpy as np

from backend.science.core import AnalysisFrame
from backend.services.vlm import get_vlm_engine, StubEngine, describe_vlm_configuration
from backend.services.costs import log_vlm_usage

logger = logging.getLogger(__name__)


class SemanticTagAnalyzer:
    """Semantic VLM analyzer for style.* and spatial.room_function.*."""

    name = "semantic_tags_vlm"

    # Single, explicit prompt that asks for normalized scores in [0, 1].
    PROMPT = (
        "You are an environmental psychology and interior-architecture expert.\n"
        "Given a single photograph of an architectural or interior space, estimate:\n"
        "1) The strength of several interior design styles (0.0–1.0 each).\n"
        "2) The plausibility that the scene serves specific room functions (0.0–1.0 each).\n\n"
        "Return ONLY strict JSON with these keys (all floats between 0.0 and 1.0):\n"
        "{\n"
        '  "style_modern": float,\n'
        '  "style_traditional": float,\n'
        '  "style_minimalist": float,\n'
        '  "style_scandinavian": float,\n'
        '  "style_industrial": float,\n'
        '  "style_rustic": float,\n'
        '  "style_bohemian": float,\n'
        '  "style_farmhouse": float,\n'
        '  "style_japandi": float,\n'
        '  "room_function_living_room": float,\n'
        '  "room_function_kitchen": float,\n'
        '  "room_function_bedroom": float,\n'
        '  "room_function_home_office": float,\n'
        '  "room_function_bathroom": float\n'
        "}\n"
    )

    def __init__(self) -> None:
        # Reserved for future versioning if we evolve the prompt.
        self.prompt_version = "1.0"

    def analyze(self, frame: AnalysisFrame) -> None:
        """Run VLM-based semantic analysis on the given frame.

        This intentionally mirrors CognitiveStateAnalyzer:
        - Uses the in-memory RGB image (np.ndarray) from the pipeline.
        - Short-circuits safely when no VLM is configured (StubEngine).
        - Logs cost once per successful real VLM call.
        """
        try:
            img = frame.original_image
            if img is None:
                logger.warning("SemanticTagAnalyzer: frame has no original_image; skipping.")
                return

            if not isinstance(img, np.ndarray):
                logger.warning(
                    "SemanticTagAnalyzer: unsupported image type %r; skipping.", type(img)
                )
                return

            ok, buffer = cv2.imencode(".jpg", img)
            if not ok:
                logger.error("SemanticTagAnalyzer: JPEG encoding failed; skipping.")
                return
            image_bytes = buffer.tobytes()

            engine = get_vlm_engine()
            prompt = self.PROMPT
            result: Dict[str, Any] = engine.analyze_image(image_bytes, prompt)

            # Stub / classroom path: record metadata only, no numeric attributes.
            if isinstance(engine, StubEngine) or result.get("stub"):
                frame.metadata["semantics.vlm"] = {
                    "status": "stub",
                    "prompt_version": self.prompt_version,
                    "engine": type(engine).__name__,
                }
                logger.info(
                    "SemanticTagAnalyzer: running in STUB mode; "
                    "no style.* or spatial.room_function.* attributes will be written."
                )
                return

            # Real-data path: record a single cost entry for this VLM call.
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
                        "source": "science_pipeline_semantic_tags",
                        "image_id": getattr(frame, "image_id", None),
                        "cost_per_1k_images_usd": cost_per_1k,
                    },
                )
            except Exception:
                # Cost logging must never break the analysis.
                logger.debug("SemanticTagAnalyzer: cost logging failed", exc_info=True)

            # Map JSON keys to canonical feature keys.
            style_map = {
                "style_modern": "style.modern",
                "style_traditional": "style.traditional",
                "style_minimalist": "style.minimalist",
                "style_scandinavian": "style.scandinavian",
                "style_industrial": "style.industrial",
                "style_rustic": "style.rustic",
                "style_bohemian": "style.bohemian",
                "style_farmhouse": "style.farmhouse",
                "style_japandi": "style.japandi",
            }
            room_map = {
                "room_function_living_room": "spatial.room_function.living_room",
                "room_function_kitchen": "spatial.room_function.kitchen",
                "room_function_bedroom": "spatial.room_function.bedroom",
                "room_function_home_office": "spatial.room_function.home_office",
                "room_function_bathroom": "spatial.room_function.bathroom",
            }

            def _clamp(x: float) -> float:
                try:
                    v = float(x)
                except Exception:
                    return 0.0
                if v < 0.0:
                    return 0.0
                if v > 1.0:
                    return 1.0
                return v

            style_scores: Dict[str, float] = {}
            room_scores: Dict[str, float] = {}

            # Emit style.* attributes.
            for json_key, feature_key in style_map.items():
                if json_key in result:
                    value = _clamp(result[json_key])
                    style_scores[feature_key] = value
                    frame.add_attribute(feature_key, value, confidence=0.85)

            # Emit spatial.room_function.* attributes.
            for json_key, feature_key in room_map.items():
                if json_key in result:
                    value = _clamp(result[json_key])
                    room_scores[feature_key] = value
                    frame.add_attribute(feature_key, value, confidence=0.9)

            # Attach primary guesses and raw payload for later inspection.
            def _argmax(mapping: Dict[str, float]) -> Dict[str, Any]:
                if not mapping:
                    return {}
                best_key = max(mapping, key=lambda k: mapping[k])
                return {"key": best_key, "score": mapping[best_key]}

            frame.metadata.setdefault("semantics", {})
            frame.metadata["semantics"]["primary_style"] = _argmax(style_scores)
            frame.metadata["semantics"]["primary_room_function"] = _argmax(room_scores)
            frame.metadata["semantics"]["raw_vlm_result"] = result
            frame.metadata["semantics"]["prompt_version"] = self.prompt_version

        except Exception as exc:
            # Fail soft: record an error flag and keep the rest of the pipeline alive.
            logger.error("SemanticTagAnalyzer failed: %s", exc)
            frame.metadata.setdefault("semantics", {})
            frame.metadata["semantics"]["error"] = str(exc)
