"""Perceptual color analysis in CIELAB space."""

from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from backend.science.core import AnalysisFrame


class ColorAnalyzer:
    """Extract color descriptors aligned with perceptual space.

    All outputs are normalized to [0, 1] where possible.
    """

    def __init__(self, max_samples: int = 4096, random_seed: int = 42):
        self.max_samples = max_samples
        self.random_seed = random_seed

    def analyze(self, frame: AnalysisFrame) -> None:
        lab = frame.ensure_lab()

        # L* channel in [0, 100]
        L = lab[:, :, 0]
        a = lab[:, :, 1]
        b = lab[:, :, 2]

        # 1. Perceptual lightness (mean L* scaled to [0, 1])
        mean_L = float(np.mean(L))
        lightness = np.clip(mean_L / 100.0, 0.0, 1.0)

        # 2. Lightness contrast (std L* normalized by a nominal max, e.g. 25)
        std_L = float(np.std(L))
        lightness_contrast = np.clip(std_L / 25.0, 0.0, 1.0)

        # 3. Warm–cool index: map mean a* and b* to [0, 1]
        mean_a = float(np.mean(a))
        mean_b = float(np.mean(b))
        warm_radius = np.sqrt(mean_a ** 2 + mean_b ** 2)
        warm_radius_norm = np.clip(warm_radius / 60.0, 0.0, 1.0)
        warm_sign = 0.0
        denom = abs(mean_a) + abs(mean_b)
        if warm_radius > 1e-6 and denom > 1e-6:
            warm_sign = (mean_a + mean_b) / denom
        warm_sign_norm = (warm_sign + 1.0) / 2.0
        warmth_index = 0.5 * warm_radius_norm + 0.5 * warm_sign_norm

        # 4. Gamut volume in a–b plane via convex hull on a subsample
        h, w = a.shape
        ab = np.stack([a.reshape(-1), b.reshape(-1)], axis=1)

        if ab.shape[0] > self.max_samples:
            rng = np.random.default_rng(self.random_seed)
            idx = rng.choice(ab.shape[0], size=self.max_samples, replace=False)
            ab_sample = ab[idx]
        else:
            ab_sample = ab

        try:
            hull = ConvexHull(ab_sample)
            raw_area = float(hull.volume)
            gamut_volume = np.clip(raw_area / (200.0 * 200.0), 0.0, 1.0)
        except Exception:
            gamut_volume = 0.0

        frame.set_attributes(
            {
                "color.lightness_mean": lightness,
                "color.lightness_contrast": lightness_contrast,
                "color.warmth_index": warmth_index,
                "color.gamut_volume": gamut_volume,
            }
        )