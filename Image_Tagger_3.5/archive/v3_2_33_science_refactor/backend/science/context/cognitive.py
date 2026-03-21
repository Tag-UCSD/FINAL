"""Cognitive state / VLM-based attribute analysis."""

from __future__ import annotations

from typing import Dict, Optional, Protocol
import numpy as np

from backend.science.core import AnalysisFrame


class PerceptionEngine(Protocol):
    """Protocol for a VLM-style perception engine.

    Implementations should accept an RGB image (H, W, 3) uint8 array and
    return a mapping from attribute keys to floats in [0, 1].
    """

    def __call__(self, image: np.ndarray) -> Dict[str, float]:  # pragma: no cover
        raise NotImplementedError


class CognitiveStateAnalyzer:
    """Bridge to VLM-based cognitive/affective estimates.

    For now this is intentionally conservative: if no perception engine
    is configured, we produce neutral 0.5 scores with explicit keys.
    """

    def __init__(self, engine: Optional[PerceptionEngine] = None):
        self.engine = engine

    def analyze(self, frame: AnalysisFrame) -> None:
        if self.engine is None:
            neutral = {
                "cog.tranquility": 0.5,
                "cog.arousal": 0.5,
                "cog.formality": 0.5,
            }
            frame.set_attributes(neutral)
            frame.metadata.setdefault("cognitive", {})["source"] = "neutral_baseline"
            return

        try:
            preds = self.engine(frame.original_image)
        except Exception:
            frame.metadata.setdefault("cognitive", {})["source"] = "engine_error"
            return

        if not isinstance(preds, dict):
            return

        clean: Dict[str, float] = {}
        for k, v in preds.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                clean[k] = float(v)
        if clean:
            frame.set_attributes(clean)
            frame.metadata.setdefault("cognitive", {})["source"] = "engine"