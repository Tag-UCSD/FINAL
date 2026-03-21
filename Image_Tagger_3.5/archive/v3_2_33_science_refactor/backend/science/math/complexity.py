"""Image complexity metrics: Shannon entropy, spatial entropy, edge density."""

from __future__ import annotations

import numpy as np
from scipy.stats import entropy
from skimage.feature import greycomatrix
from backend.science.core import AnalysisFrame


class ComplexityAnalyzer:
    """Compute basic complexity metrics on gray image and edges.

    All outputs are normalized roughly into [0, 1] for downstream BN use.
    """

    def __init__(self, glcm_levels: int = 32, glcm_downsample_max: int = 256):
        self.glcm_levels = glcm_levels
        self.glcm_downsample_max = glcm_downsample_max

    def analyze(self, frame: AnalysisFrame) -> None:
        gray = frame.ensure_gray().astype(np.float32)
        edges = frame.ensure_edges()

        # Normalize gray to 0..1
        if gray.max() > 0:
            gray_norm = gray / float(gray.max())
        else:
            gray_norm = gray

        # 1. Shannon entropy of gray histogram
        hist, _ = np.histogram(gray_norm, bins=64, range=(0.0, 1.0))
        if hist.sum() > 0:
            hist = hist.astype(np.float32) / float(hist.sum())
            shannon = float(entropy(hist, base=2))
            shannon_norm = np.clip(shannon / np.log2(64.0), 0.0, 1.0)
        else:
            shannon_norm = 0.0

        # 2. Spatial entropy using GLCM on downsampled image
        spatial_entropy_norm = self._spatial_entropy(gray_norm)

        # 3. Edge density
        edge_density = float((edges > 0).sum()) / float(edges.size) if edges.size > 0 else 0.0
        edge_density = float(np.clip(edge_density, 0.0, 1.0))

        frame.set_attributes(
            {
                "complexity.shannon_entropy": shannon_norm,
                "complexity.spatial_entropy": spatial_entropy_norm,
                "complexity.edge_density": edge_density,
            }
        )

    def _spatial_entropy(self, gray_norm: np.ndarray) -> float:
        h, w = gray_norm.shape
        # Downsample if needed to keep GLCM manageable
        factor = max(h, w) / float(self.glcm_downsample_max)
        if factor > 1.0:
            new_h = max(1, int(h / factor))
            new_w = max(1, int(w / factor))
            gray_ds = gray_norm[0:h:new_h, 0:w:new_w]
        else:
            gray_ds = gray_norm

        if gray_ds.size < 4:
            return 0.0

        # Quantize
        bins = self.glcm_levels
        quant = np.clip((gray_ds * (bins - 1)).astype(np.int32), 0, bins - 1)

        glcm = greycomatrix(
            quant,
            distances=[1],
            angles=[0.0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
            levels=bins,
            symmetric=True,
            normed=True,
        )

        # Joint distribution over gray-level pairs
        P = glcm[:, :, 0, :]  # shape (bins, bins, angles)
        P = P.mean(axis=-1)    # average over angles
        P_flat = P.reshape(-1)
        P_flat = P_flat[P_flat > 0]
        if P_flat.size == 0:
            return 0.0
        H = float(entropy(P_flat, base=2))
        # maximum possible entropy is log2(bins^2)
        H_max = 2.0 * np.log2(float(bins))
        return float(np.clip(H / H_max, 0.0, 1.0))