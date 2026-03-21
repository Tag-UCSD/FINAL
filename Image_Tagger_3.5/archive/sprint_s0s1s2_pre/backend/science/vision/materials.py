"""
Material analysis module for v3 Science Pipeline.

Ported heuristics from v2.6.3 MaterialClassifier into the v3
AnalysisFrame pattern. This module is deliberately lightweight and
does not depend on the v2 AttributeExtractor base class.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import cv2

from backend.science.core import AnalysisFrame
from backend.services.vlm import get_vlm_engine, StubEngine


class MaterialAnalyzer:
    """
    Heuristic-based material classification.

    Uses simple HSV and luminance/texture rules to estimate coverage
    of wood, metal, and glass in the scene. Values are normalized
    coverage ratios in [0, 1].
    """

    @staticmethod
    def analyze(frame: AnalysisFrame) -> None:
        img = frame.original_image
        if img is None:
            return

        # Convert to uint8 RGB for OpenCV operations
        if img.dtype == np.float32 or img.dtype == np.float64:
            image_uint8 = (img * 255).astype(np.uint8)
        else:
            image_uint8 = img

        # Convert to HSV for material heuristics
        hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV)

        # --- Wood heuristic (ported from v2 logic) ---
        # Brown-ish hues (approx 5–30 in OpenCV HSV),
        # with reasonable saturation and value.
        wood_mask = (
            (hsv[:, :, 0] >= 5)
            & (hsv[:, :, 0] <= 30)
            & (hsv[:, :, 1] > 30)
            & (hsv[:, :, 2] > 50)
        )
        wood_coverage = float(np.sum(wood_mask) / wood_mask.size)
        frame.add_attribute("material.wood_coverage", wood_coverage)

        # --- Metal heuristic (simplified from v2) ---
        # Low saturation, mid-to-high value → shiny / metallic regions.
        metal_mask = (hsv[:, :, 1] < 30) & (hsv[:, :, 2] > 150)
        metal_coverage = float(np.sum(metal_mask) / metal_mask.size)
        frame.add_attribute("material.metal_coverage", metal_coverage)

        # --- Glass heuristic (ported from v2) ---
        # High luminance + low local variance → smooth bright panes.
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        bright_mask = gray > 200

        # Local variance proxy using a smoothing kernel
        kernel = np.ones((5, 5), np.float32) / 25.0
        local_mean = cv2.filter2D(gray.astype(float), -1, kernel)
        local_var = (gray.astype(float) - local_mean) ** 2
        smooth_mask = local_var < 100.0

        glass_mask = bright_mask & smooth_mask
        glass_coverage = float(np.sum(glass_mask) / glass_mask.size)
        frame.add_attribute("material.glass_coverage", glass_coverage)
        # --- L1 material cues (read-only; support higher tiers) ---
        # Normalized brightness (0-1) from grayscale.
        gray_float = gray.astype(float) / 255.0
        brightness_mean = float(gray_float.mean())
        frame.add_attribute("materials.cues.brightness_mean", brightness_mean, confidence=0.7)

        # Global texture variance (roughness proxy).
        texture_variance = float(gray_float.var())
        frame.add_attribute("materials.cues.texture_variance", min(texture_variance * 10.0, 1.0), confidence=0.6)

        # Mean saturation and value in HSV.
        sat_mean = float(hsv[:, :, 1].mean() / 255.0)
        val_mean = float(hsv[:, :, 2].mean() / 255.0)
        frame.add_attribute("materials.cues.saturation_mean", sat_mean, confidence=0.7)
        frame.add_attribute("materials.cues.value_mean", val_mean, confidence=0.7)

        # Specularity proxy: proportion of high-value, low-saturation pixels.
        spec_mask = (hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 200)
        specularity_proxy = float(spec_mask.sum() / spec_mask.size)
        frame.add_attribute("materials.cues.specularity_proxy", specularity_proxy, confidence=0.6)

        # --- Substrate heuristics beyond wood/metal/glass ---
        # Stone/Concrete: low saturation, mid value, higher roughness.
        stone_mask = (
            (hsv[:, :, 1] < 60) &
            (hsv[:, :, 2] > 60) &
            (hsv[:, :, 2] < 200)
        )
        stone_coverage = float(stone_mask.sum() / stone_mask.size)
        frame.add_attribute("materials.substrate.stone_concrete", stone_coverage, confidence=0.5)

        # Plaster/Gypsum: very low saturation, high value, low variance.
        plaster_mask = (
            (hsv[:, :, 1] < 30) &
            (hsv[:, :, 2] > 180)
        )
        plaster_coverage = float(plaster_mask.sum() / plaster_mask.size)
        frame.add_attribute("materials.substrate.plaster_gypsum", plaster_coverage, confidence=0.5)

        # Tile/Ceramic: bright and moderately saturated with elevated local variance.
        # We reuse local_var from the glass heuristic as a crude texture cue.
        tile_mask = (
            (hsv[:, :, 2] > 150) &
            (hsv[:, :, 1] > 40) &
            (local_var > 50.0)
        )
        tile_coverage = float(tile_mask.sum() / tile_mask.size)
        frame.add_attribute("materials.substrate.tile_ceramic", tile_coverage, confidence=0.4)


def _maybe_run_materials_vlm(frame: AnalysisFrame, image_uint8: np.ndarray) -> None:
    """Optional VLM pass for materials.

    This function is intentionally conservative: it never writes numeric
    materials.* attributes. Instead, it records a structured candidate list
    under frame.metadata["materials.vlm_candidates"] when a real VLM is configured.

    When running under StubEngine, we record only a note so downstream science
    knows that no VLM data were present.
    """
    try:
        ok, buffer = cv2.imencode(".jpg", image_uint8)
        if not ok:
            return
        image_bytes = buffer.tobytes()
        engine = get_vlm_engine()

        prompt = (
            "You are an architectural materials analyst. "
            "Given this interior image, estimate the presence of the following materials substrates, "
            "each from 0.0 to 1.0, and provide a brief evidence string. "
            "Substrates:
"
            "- materials.substrate.stone_concrete
"
            "- materials.substrate.plaster_gypsum
"
            "- materials.substrate.tile_ceramic
"
            "Return STRICT JSON as a list of objects with fields "
            "{'key': <substrate_key>, 'present': <0-1>, 'confidence': <0-1>, 'evidence': <short text>}."
        )
        result = engine.analyze_image(image_bytes, prompt)
    except Exception:
        # We silently skip VLM errors for materials; core CV cues remain available.
        return

    # Stub / classroom path.
    if isinstance(engine, StubEngine) or (isinstance(result, dict) and result.get("stub")):
        frame.metadata["materials.vlm_candidates"] = {
            "note": "VLM in stub mode; no materials.* VLM candidates.",
            "engine": type(engine).__name__,
        }
        return

    candidates = []
    if isinstance(result, list):
        candidates = result
    elif isinstance(result, dict) and "materials" in result and isinstance(result["materials"], list):
        candidates = result["materials"]

    frame.metadata["materials.vlm_candidates"] = {
        "engine": type(engine).__name__,
        "candidates": candidates,
    }
