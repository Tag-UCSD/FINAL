"""
Regional (patchwise) spatial frequency analysis.

This module slices the grayscale image into patches and computes
band-limited power per patch, then summarizes their distribution. It is
designed as a forward-looking scaffold for more advanced multiscale,
oriented frequency analysis.
"""

import numpy as np
from backend.science.core import AnalysisFrame
from backend.science.contracts import fail


class RegionalSpatialFrequencyAnalyzer:
    name = "regional_spatial_frequency"
    tier = "L0"
    requires = ["gray_image"]
    provides = [
        "spatial_freq_reg.low_mean",
        "spatial_freq_reg.low_var",
        "spatial_freq_reg.mid_mean",
        "spatial_freq_reg.mid_var",
        "spatial_freq_reg.high_mean",
        "spatial_freq_reg.high_var",
    ]

    def __init__(self, patch: int = 64, stride: int | None = None) -> None:
        self.patch = patch
        self.stride = stride or patch

    def _band_powers_fft(self, patch: np.ndarray) -> tuple[float, float, float]:
        f = np.fft.fft2(patch.astype(np.float32))
        fshift = np.fft.fftshift(f)
        ps = np.abs(fshift) ** 2
        h, w = ps.shape
        cy, cx = h // 2, w // 2
        y, x = np.indices((h, w))
        r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2)
        r = r / r.max()

        low_mask = r < 0.33
        mid_mask = (r >= 0.33) & (r < 0.66)
        high_mask = r >= 0.66

        low = float(ps[low_mask].mean()) if np.any(low_mask) else 0.0
        mid = float(ps[mid_mask].mean()) if np.any(mid_mask) else 0.0
        high = float(ps[high_mask].mean()) if np.any(high_mask) else 0.0
        return low, mid, high

    def analyze(self, frame: AnalysisFrame) -> None:
        gray = frame.gray_image
        if gray is None:
            fail(frame, self.name, "missing gray_image")
            return

        H, W = gray.shape
        p, s = self.patch, self.stride
        lows, mids, highs = [], [], []

        for y in range(0, max(H - p + 1, 1), s):
            if y + p > H:
                break
            for x in range(0, max(W - p + 1, 1), s):
                if x + p > W:
                    break
                patch = gray[y : y + p, x : x + p]
                low, mid, high = self._band_powers_fft(patch)
                lows.append(low)
                mids.append(mid)
                highs.append(high)

        if not lows:
            fail(frame, self.name, "image too small for regional analysis")
            return

        lows = np.array(lows, dtype=np.float32)
        mids = np.array(mids, dtype=np.float32)
        highs = np.array(highs, dtype=np.float32)

        frame.add_attribute("spatial_freq_reg.low_mean", float(lows.mean()))
        frame.add_attribute("spatial_freq_reg.low_var", float(lows.var()))
        frame.add_attribute("spatial_freq_reg.mid_mean", float(mids.mean()))
        frame.add_attribute("spatial_freq_reg.mid_var", float(mids.var()))
        frame.add_attribute("spatial_freq_reg.high_mean", float(highs.mean()))
        frame.add_attribute("spatial_freq_reg.high_var", float(highs.var()))
