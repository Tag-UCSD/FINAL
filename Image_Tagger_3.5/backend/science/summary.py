"""
High-level science composites for Image Tagger v3.3.

This module defines ScienceSummaryAnalyzer, which computes:

  * science.visual_richness        (0.0–1.0)
  * science.organized_complexity   (0.0–1.0)
  * science.visual_richness_bin    (0=low, 1=mid, 2=high)
  * science.organized_complexity_bin (0=low, 1=mid, 2=high)

It is designed to sit on top of the lower-level analyzers in
backend.science.math and backend.science.spatial.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Dict, Any, Optional

import math

from backend.science.core import AnalysisFrame


COLOR_KEYS = []
COMPLEXITY_KEYS = []
TEXTURE_KEYS = ['texture.macro.homogeneity', 'texture.micro.contrast', 'texture.micro.homogeneity', 'texture.macro.contrast']
FRACTAL_KEYS = ['fractal.D']


def _safe_avg(frame: AnalysisFrame, keys: Sequence[str]) -> Optional[float]:
    values = [frame.attributes[k] for k in keys if k in frame.attributes]
    if not values:
        return None
    return float(sum(values) / len(values))


def _clamp01(x: float) -> float:
    if math.isnan(x):
        return 0.0
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def _to_bin(x: float) -> int:
    if x < 0.33:
        return 0
    if x < 0.66:
        return 1
    return 2


@dataclass
class ScienceSummaryAnalyzer:
    """
    Lightweight composite index builder.

    The goal is not to be "the final word" on the science, but to
    provide stable, BN-friendly scalars and discrete bins.
    """

    def analyze(self, frame: AnalysisFrame) -> None:
        # 1. Visual richness: color + texture + a complexity touch.
        color   = _safe_avg(frame, COLOR_KEYS)
        texture = _safe_avg(frame, TEXTURE_KEYS)
        comp    = _safe_avg(frame, COMPLEXITY_KEYS)

        components = [v for v in (color, texture, comp) if v is not None]
        if components:
            raw_vr = sum(components) / len(components)
            vr = _clamp01(raw_vr)
            frame.add_attribute("science.visual_richness", vr, confidence=1.0)
            frame.add_attribute("science.visual_richness_bin", float(_to_bin(vr)), confidence=1.0)

        # 2. Organized complexity: complexity + fractals (if any).
        comp2 = _safe_avg(frame, COMPLEXITY_KEYS)
        frac  = _safe_avg(frame, FRACTAL_KEYS)

        components2 = [v for v in (comp2, frac) if v is not None]
        if components2:
            raw_oc = sum(components2) / len(components2)
            oc = _clamp01(raw_oc)
            frame.add_attribute("science.organized_complexity", oc, confidence=1.0)
            frame.add_attribute("science.organized_complexity_bin", float(_to_bin(oc)), confidence=1.0)
