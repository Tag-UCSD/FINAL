"""
VLM-assisted architectural pattern detection (stubbed).

This module defines the prompt and schema for architectural patterns
(e.g., prospect/refuge, circulation), but deliberately avoids making
strong numeric commitments when no real VLM is configured.
"""

from __future__ import annotations

from typing import Any, Dict, List

from backend.science.core import AnalysisFrame
from backend.science.contracts import fail
from backend.science.semantics.ontology import ARCH_PARTS  # noqa: F401  # placeholder for future cross-linking
from backend.services.vlm import get_vlm_engine, StubEngine, describe_vlm_configuration
from backend.services.costs import log_vlm_usage
import cv2
import numpy as np


ARCH_PATTERNS: Dict[str, str] = {
    "arch.pattern.prospect_strong": "Strong outward views and long sightlines (Prospect).",
    "arch.pattern.refuge_strong": "Strong sense of enclosure/refuge via partial enclosure.",
    "arch.pattern.refuge_nook": "Local nook/alcove suitable for retreat.",
    "arch.pattern.axial_circulation_clear": "Clear axial circulation path.",
    "arch.pattern.circulation_maze_like": "Maze-like, complex circulation.",
    "arch.pattern.double_height_space": "Double-height or tall volume.",
    "arch.pattern.corner_window": "Corner window or wrapping glazing.",
    "arch.pattern.perimeter_seating": "Seating arranged along the room perimeter.",
    "arch.pattern.central_hearth": "Fireplace or central hearth as focal point.",
    "arch.pattern.gallery_edge": "Balcony or gallery overlooking lower space.",
    "arch.pattern.daylight_soft": "Soft diffuse daylight with low glare.",
    "arch.pattern.daylight_hard": "Strong direct daylight with sharp shadows.",
    "arch.pattern.skylight_dominant": "Skylights as primary daylight source.",
    "arch.pattern.threshold_emphasized": "Emphasized entry threshold or portal.",
    "arch.pattern.colonnade": "Series of columns forming a colonnade.",
    "arch.pattern.bay_window": "Projecting bay window.",
    "arch.pattern.staircase_sculptural": "Staircase acting as sculptural feature.",
    "arch.pattern.long_view_corridor": "Long view along a corridor.",
    "arch.pattern.loft_mezzanine": "Loft or mezzanine overlooking space.",
    "arch.pattern.window_seat_niche": "Window seat or deep sill niche.",
}
# Subset of patterns for which we are willing to emit numeric attributes
# in this initial VLM wiring sprint. All others remain metadata-only.
ACTIVE_PATTERN_KEYS = {
    "arch.pattern.prospect_strong",
    "arch.pattern.refuge_strong",
    "arch.pattern.daylight_soft",
    "arch.pattern.daylight_hard",
    "arch.pattern.double_height_space",
}



class ArchPatternsVLMAnalyzer:
    name = "arch_patterns_vlm"
    tier = "L4"
    requires = ["image_url"]
    provides = ["arch.patterns.candidates"]

    def __init__(self, prompt_version: str = "arch_patterns_v1") -> None:
        self.prompt_version = prompt_version

    def build_prompt(self) -> str:
        items = []
        for key, desc in sorted(ARCH_PATTERNS.items()):
            items.append(f"- {key}: {desc}")
        patterns_block = "\n".join(items)
        return (
            "You are an architectural cognition assistant. "
            "Given this interior image, estimate the presence of the following architectural patterns.\n"
            f"{patterns_block}\n"
            "Return STRICT JSON as a list of objects with fields "
            "{'key': <pattern_key>, 'present': <0-1>, 'confidence': <0-1>, 'evidence': <short text>}."
        )

    def analyze(self, frame: AnalysisFrame) -> None:
        """Run VLM-based pattern analysis on the given frame.

        This mirrors the CognitiveStateAnalyzer pattern:
        - If running with StubEngine (or stub result), we only record metadata.
        - If running with a real VLM, we emit numeric attributes for a small,
          carefully chosen subset of patterns (ACTIVE_PATTERN_KEYS) and
          keep the full candidate list in metadata.
        """
        # We prefer to work from the in-memory RGB image, not the URL.
        img = frame.original_image
        if img is None:
            fail(frame, self.name, "no original_image available for VLM encoding")
            return

        if not isinstance(img, np.ndarray):
            fail(frame, self.name, f"unsupported image type for VLM: {type(img)}")
            return

        ok, buffer = cv2.imencode(".jpg", img)
        if not ok:
            fail(frame, self.name, "JPEG encoding failed for VLM.")
            return
        image_bytes = buffer.tobytes()

        prompt = self.build_prompt()
        engine = get_vlm_engine()
        try:
            result: Any = engine.analyze_image(image_bytes, prompt)
        except Exception as exc:  # pragma: no cover - network
            fail(frame, self.name, f"VLM error: {exc}")
            return

        # Stub / classroom path: record metadata only, no numeric priors.
        if isinstance(engine, StubEngine) or (isinstance(result, dict) and result.get("stub")):
            frame.metadata["arch.patterns.candidates"] = {
                "prompt": prompt,
                "prompt_version": self.prompt_version,
                "candidates": [],
                "engine": type(engine).__name__,
                "note": "VLM running in stub mode; no numeric arch.pattern.* attributes emitted.",
            }
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
                    "source": "science_pipeline_arch_patterns",
                    "image_id": getattr(frame, "image_id", None),
                    "cost_per_1k_images_usd": cost_per_1k,
                },
            )
        except Exception:
            # Cost logging must never break the analysis.
            pass


        # Normalize result into a list of candidate dicts.
        candidates: List[Dict[str, Any]]
        if isinstance(result, list):
            candidates = result
        elif isinstance(result, dict) and "patterns" in result:
            patterns_val = result.get("patterns")
            if isinstance(patterns_val, list):
                candidates = patterns_val
            else:
                candidates = []
        else:
            # Unexpected shape; record and bail without emitting numerics.
            frame.metadata["arch.patterns.candidates"] = {
                "prompt": prompt,
                "prompt_version": self.prompt_version,
                "raw_result": result,
                "engine": type(engine).__name__,
                "note": "VLM returned unexpected JSON shape; no numeric arch.pattern.* attributes emitted.",
            }
            return

        def _clamp(x: Any) -> float:
            try:
                v = float(x)
            except Exception:
                return 0.0
            if v < 0.0:
                return 0.0
            if v > 1.0:
                return 1.0
            return v

        # Emit numeric attributes only for a small, high-value subset of patterns.
        for cand in candidates:
            if not isinstance(cand, dict):
                continue
            key = cand.get("key")
            if not isinstance(key, str):
                continue
            if key not in ACTIVE_PATTERN_KEYS:
                continue

            present = _clamp(cand.get("present", cand.get("value", 0.0)))
            confidence = _clamp(cand.get("confidence", 0.7))
            evidence = cand.get("evidence") or cand.get("reason") or ""

            # Use explicit string keys so our governance tests can see coverage.
            if key == "arch.pattern.prospect_strong":
                frame.add_attribute("arch.pattern.prospect_strong", present, confidence=confidence)
                attr_key = "arch.pattern.prospect_strong"
            elif key == "arch.pattern.refuge_strong":
                frame.add_attribute("arch.pattern.refuge_strong", present, confidence=confidence)
                attr_key = "arch.pattern.refuge_strong"
            elif key == "arch.pattern.daylight_soft":
                frame.add_attribute("arch.pattern.daylight_soft", present, confidence=confidence)
                attr_key = "arch.pattern.daylight_soft"
            elif key == "arch.pattern.daylight_hard":
                frame.add_attribute("arch.pattern.daylight_hard", present, confidence=confidence)
                attr_key = "arch.pattern.daylight_hard"
            elif key == "arch.pattern.double_height_space":
                frame.add_attribute("arch.pattern.double_height_space", present, confidence=confidence)
                attr_key = "arch.pattern.double_height_space"
            else:
                # Any other keys (even if in ACTIVE_PATTERN_KEYS) are ignored for now.
                continue

            # Attach evidence to metadata for inspection in the Feature Navigator / Admin tools.
            meta = frame.metadata.get(attr_key, {})
            meta["evidence"] = str(evidence)
            meta["source"] = "vlm"
            meta["engine"] = type(engine).__name__
            frame.metadata[attr_key] = meta

        # Always keep the full candidate list for later inspection.
        frame.metadata["arch.patterns.candidates"] = {
            "prompt": prompt,
            "prompt_version": self.prompt_version,
            "candidates": candidates,
            "engine": type(engine).__name__,
        }