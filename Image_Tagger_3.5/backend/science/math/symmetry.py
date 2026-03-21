"""backend.science.math.symmetry

Primary function:
  Compute simple bilateral symmetry proxies for an AnalysisFrame.

Inputs:
  frame: AnalysisFrame with grayscale_image (H,W) float/uint8.

Outputs (stored into frame.metrics):
  symmetry.vertical_score  : float in [0,1], higher = more left-right symmetry.
  symmetry.horizontal_score: float in [0,1], higher = more top-bottom symmetry.
  symmetry.mean_score      : float in [0,1].

Notes:
  This is a deterministic, low-cost proxy intended for teaching and first-pass science.
  It measures pixelwise correlation between an image half and its mirror.
"""

from __future__ import annotations

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None  # type: ignore

class SymmetryAnalyzer:
    """Bilateral symmetry proxies."""

    def analyze(self, frame) -> None:
        gray = getattr(frame, "grayscale_image", None)
        if gray is None:
            return
        img = np.asarray(gray).astype(np.float32)
        if img.ndim != 2:
            img = img.mean(axis=-1)

        h, w = img.shape
        if h < 4 or w < 4:
            return

        # Normalize to zero-mean, unit-variance for stability.
        mu = img.mean()
        sigma = img.std() + 1e-6
        z = (img - mu) / sigma

        # Vertical symmetry (left-right)
        left = z[:, : w // 2]
        right = z[:, w - left.shape[1] :][:, ::-1]
        v_score = self._corr(left, right)

        # Horizontal symmetry (top-bottom)
        top = z[: h // 2, :]
        bottom = z[h - top.shape[0] :, :][::-1, :]
        h_score = self._corr(top, bottom)

        mean_score = float(np.nanmean([v_score, h_score]))

        frame.metrics["symmetry.vertical_score"] = float(v_score)
        frame.metrics["symmetry.horizontal_score"] = float(h_score)
        frame.metrics["symmetry.mean_score"] = mean_score

    @staticmethod
    def _corr(a: np.ndarray, b: np.ndarray) -> float:
        a_f = a.reshape(-1)
        b_f = b.reshape(-1)
        if a_f.size == 0 or b_f.size == 0:
            return 0.0
        denom = (np.linalg.norm(a_f) * np.linalg.norm(b_f)) + 1e-6
        if denom == 0:
            return 0.0
        c = float(np.dot(a_f, b_f) / denom)
        # map from [-1,1] to [0,1]
        return max(0.0, min(1.0, (c + 1.0) / 2.0))
