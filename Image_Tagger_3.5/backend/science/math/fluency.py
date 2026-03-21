"""backend.science.math.fluency

Primary function:
  Compute a lightweight perceptual-fluency proxy.

Inputs:
  frame: AnalysisFrame with complexity + texture results already in metrics.

Outputs (stored into frame.metrics):
  fluency.score : float in [0,1], higher = easier to perceptually parse.

Notes:
  This combines low edge density, moderate entropy, and texture regularity.
  It is a heuristic starting point suitable for early experiments and teaching.
"""

from __future__ import annotations

import numpy as np


class FluencyAnalyzer:
    def analyze(self, frame) -> None:
        m = getattr(frame, "metrics", {}) or {}
        edge_density = float(m.get("complexity.edge_density", np.nan))
        entropy = float(m.get("complexity.shannon_entropy", np.nan))
        glcm_contrast = float(m.get("texture.glcm_contrast_mean", np.nan))

        # Normalize inputs to rough [0,1] scales with conservative bounds.
        # Missing values propagate to nan then to fallback neutral score.
        def clamp01(x):
            return float(np.clip(x, 0.0, 1.0))

        ed_n = clamp01(edge_density / 0.25) if np.isfinite(edge_density) else np.nan  # 0.25 is "busy"
        ent_n = clamp01(entropy / 8.0) if np.isfinite(entropy) else np.nan           # entropy ~0..8 for 256 bins
        con_n = clamp01(glcm_contrast / 10.0) if np.isfinite(glcm_contrast) else np.nan

        # Fluency rises when edges and contrast are low/moderate and entropy is not extreme.
        components = []
        if np.isfinite(ed_n):
            components.append(1.0 - ed_n)
        if np.isfinite(ent_n):
            components.append(1.0 - abs(ent_n - 0.5) * 2.0)  # best around mid-entropy
        if np.isfinite(con_n):
            components.append(1.0 - con_n)

        if not components:
            score = 0.5
        else:
            score = float(np.clip(np.mean(components), 0.0, 1.0))

        frame.metrics["fluency.score"] = score
