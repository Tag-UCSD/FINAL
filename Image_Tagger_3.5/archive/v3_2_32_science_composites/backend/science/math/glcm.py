"""Texture metrics using GLCM at multiple scales."""

from __future__ import annotations

import numpy as np
from skimage.feature import greycomatrix, greycoprops
from backend.science.core import AnalysisFrame


class TextureAnalyzer:
    """Compute micro- and macro-texture metrics using GLCM.

    The outputs are aggregated across orientations and normalized for BN use.
    """

    def __init__(self, levels: int = 32, micro_distance: int = 1, macro_distance: int = 5):
        self.levels = levels
        self.micro_distance = micro_distance
        self.macro_distance = macro_distance

    def analyze(self, frame: AnalysisFrame) -> None:
        gray = frame.ensure_gray().astype(np.float32)
        if gray.size == 0:
            return

        # Normalize gray to 0..1 then quantize
        if gray.max() > 0:
            gray_norm = gray / float(gray.max())
        else:
            gray_norm = gray
        quant = np.clip((gray_norm * (self.levels - 1)).astype(np.int32), 0, self.levels - 1)

        micro = self._compute_props(quant, self.micro_distance)
        macro = self._compute_props(quant, self.macro_distance)

        frame.set_attributes(
            {
                "texture.micro.contrast": micro["contrast"],
                "texture.micro.homogeneity": micro["homogeneity"],
                "texture.micro.energy": micro["energy"],
                "texture.macro.contrast": macro["contrast"],
                "texture.macro.homogeneity": macro["homogeneity"],
                "texture.macro.energy": macro["energy"],
            }
        )

    def _compute_props(self, quant: np.ndarray, distance: int) -> dict:
        glcm = greycomatrix(
            quant,
            distances=[distance],
            angles=[0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=self.levels,
            symmetric=True,
            normed=True,
        )
        props = {}
        for name in ("contrast", "homogeneity", "energy"):
            vals = greycoprops(glcm, name)
            mean_val = float(np.mean(vals))
            # simple normalization heuristics
            if name == "contrast":
                # contrast can be large; squash via 1 - exp(-x / c)
                c = 4.0
                props[name] = float(1.0 - np.exp(-mean_val / c))
            elif name == "energy":
                # energy already in [0,1]
                props[name] = float(np.clip(mean_val, 0.0, 1.0))
            else:
                # homogeneity is [0,1]
                props[name] = float(np.clip(mean_val, 0.0, 1.0))
        return props