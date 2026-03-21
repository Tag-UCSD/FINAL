"""backend.science.math.naturalness

Primary function:
  Compute heuristic 'naturalness' proxies from color statistics.

Inputs:
  frame: AnalysisFrame with rgb_image or lab_image.

Outputs (stored into frame.metrics):
  naturalness.green_ratio   : fraction of pixels in green-hue band.
  naturalness.blue_ratio    : fraction of pixels in blue-hue band.
  naturalness.earth_ratio   : fraction of pixels in low-sat warm band.
  naturalness.score         : float in [0,1] combining the above.

Notes:
  This is intentionally lightweight. It is *not* a semantic detector of nature,
  but a perceptual proxy capturing biophilic chromatic signatures.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore


class NaturalnessAnalyzer:
    def analyze(self, frame) -> None:
        rgb = getattr(frame, "rgb_image", None)
        if rgb is None:
            return

        img = np.asarray(rgb)
        if img.ndim != 3 or img.shape[2] != 3:
            return

        # Convert to HSV for hue bands if cv2 is available; else approximate.
        if cv2 is not None:
            hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
            h = hsv[..., 0].astype(np.float32) * 2.0  # 0..360
            s = hsv[..., 1].astype(np.float32) / 255.0
            v = hsv[..., 2].astype(np.float32) / 255.0
        else:
            # crude fallback using normalized RGB
            r, g, b = img[..., 0].astype(np.float32), img[..., 1].astype(np.float32), img[..., 2].astype(np.float32)
            denom = (r + g + b + 1e-6)
            r_n, g_n, b_n = r / denom, g / denom, b / denom
            h = np.zeros_like(r_n)
            h[g_n > r_n] = 120.0
            h[b_n > g_n] = 240.0
            s = 1.0 - np.minimum.reduce([r_n, g_n, b_n])
            v = denom / denom.max()

        # Green band: 70–170 deg, adequate saturation/brightness
        green = (h >= 70) & (h <= 170) & (s >= 0.15) & (v >= 0.15)
        # Blue band: 190–260 deg
        blue = (h >= 190) & (h <= 260) & (s >= 0.12) & (v >= 0.12)
        # Earth / warm low-sat band: 15–60 deg with low saturation, medium brightness
        earth = (h >= 15) & (h <= 60) & (s <= 0.35) & (v >= 0.2)

        total = float(img.shape[0] * img.shape[1])
        green_ratio = float(green.sum() / total)
        blue_ratio = float(blue.sum() / total)
        earth_ratio = float(earth.sum() / total)

        # Simple convex combination: greens count most, then blues, then earth tones.
        score = float(np.clip(0.55 * green_ratio + 0.30 * blue_ratio + 0.15 * earth_ratio, 0.0, 1.0))

        frame.metrics["naturalness.green_ratio"] = green_ratio
        frame.metrics["naturalness.blue_ratio"] = blue_ratio
        frame.metrics["naturalness.earth_ratio"] = earth_ratio
        frame.metrics["naturalness.score"] = score
