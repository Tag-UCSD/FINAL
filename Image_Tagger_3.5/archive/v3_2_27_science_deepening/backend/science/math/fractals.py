"""Fractal dimension estimates based on box-counting on edge maps."""

from __future__ import annotations

import numpy as np
from backend.science.core import AnalysisFrame


class FractalAnalyzer:
    """Estimate fractal dimension of edge structure.

    We use a simple box-counting method over dyadic box sizes.
    """

    def __init__(self, min_box_size: int = 2):
        self.min_box_size = min_box_size

    def analyze(self, frame: AnalysisFrame) -> None:
        edges = frame.ensure_edges()
        if edges.size == 0:
            return

        # binary mask
        Z = (edges > 0).astype(np.uint8)
        d_raw = self._box_count_dim(Z)
        # For typical natural scenes, D is in [1, 2]; map linearly to [0, 1]
        if np.isnan(d_raw):
            d_norm = 0.0
        else:
            d_norm = (d_raw - 1.0) / 1.0
            d_norm = float(np.clip(d_norm, 0.0, 1.0))

        frame.set_attribute("fractal.dimension", d_norm)

    def _box_count_dim(self, Z: np.ndarray) -> float:
        # Pad to square
        h, w = Z.shape
        n = max(h, w)
        # Next power of two
        n2 = 1 << (n - 1).bit_length()
        pad_h = n2 - h
        pad_w = n2 - w
        Z_padded = np.pad(Z, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=0)

        sizes = []
        counts = []

        size = n2
        while size >= self.min_box_size:
            num = n2 // size
            if num == 0:
                break
            reshaped = Z_padded.reshape(num, size, num, size)
            reshaped = reshaped.swapaxes(1, 2)
            occupied = reshaped.max(axis=(-1, -2))
            count = int(np.count_nonzero(occupied))
            if count > 0:
                sizes.append(size)
                counts.append(count)
            size //= 2

        if len(sizes) < 2:
            return float("nan")

        sizes = np.array(sizes, dtype=np.float64)
        counts = np.array(counts, dtype=np.float64)

        logs = np.log(sizes)
        log_counts = np.log(counts)
        A = np.vstack([logs, np.ones_like(logs)]).T
        coeffs, *_ = np.linalg.lstsq(A, log_counts, rcond=None)
        slope = coeffs[0]
        return float(-slope)