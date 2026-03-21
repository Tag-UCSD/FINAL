import logging
from typing import Optional

import numpy as np

try:
    from skimage.feature import graycomatrix, graycoprops
    _HAS_SKIMAGE = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_SKIMAGE = False

from backend.science.core import AnalysisFrame

logger = logging.getLogger(__name__)


class TextureAnalyzer:
    """Compute simple GLCM-based texture metrics.

    This is deliberately conservative: we use a single distance and a small set
    of angles, then aggregate by taking the mean across orientations. The goal
    is to produce stable, interpretable scalar attributes that can feed
    downstream CNfA models.
    """

    @classmethod
    def analyze(cls, frame: AnalysisFrame) -> None:
        if frame.gray_image is None:
            frame.ensure_gray()

        gray = frame.gray_image
        if gray is None:
            # Fallback: neutral texture values
            logger.warning("TextureAnalyzer: gray_image missing; writing neutral texture attributes.")
            cls._write_neutral(frame)
            return

        # Normalize to uint8 for skimage
        try:
            arr = np.asarray(gray)
            if arr.ndim == 3:
                # Defensive: convert RGB to luminance
                arr = np.dot(arr[:, :, :3], [0.299, 0.587, 0.114])
            arr = arr.astype(np.float32)
            if arr.max() > 0:
                arr = arr / arr.max()
            arr = (arr * 255).astype(np.uint8)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("TextureAnalyzer: failed to normalize image: %s", exc)
            cls._write_neutral(frame)
            return

        if not _HAS_SKIMAGE:
            logger.warning("TextureAnalyzer: skimage not available; writing neutral texture attributes.")
            cls._write_neutral(frame)
            return

        try:
            distances = [1]
            angles = [0, np.pi / 4.0, np.pi / 2.0, 3.0 * np.pi / 4.0]
            glcm = graycomatrix(
                arr,
                distances=distances,
                angles=angles,
                levels=256,
                symmetric=True,
                normed=True,
            )

            contrast = graycoprops(glcm, "contrast").mean()
            homogeneity = graycoprops(glcm, "homogeneity").mean()
            energy = graycoprops(glcm, "energy").mean()

            # Normalize contrast into [0,1] by a simple monotonic transform
            norm_contrast = contrast / (contrast + 1.0)

            frame.add_attribute("texture.glcm.contrast", float(norm_contrast), confidence=0.9)
            frame.add_attribute("texture.glcm.homogeneity", float(homogeneity), confidence=0.9)
            frame.add_attribute("texture.glcm.energy", float(energy), confidence=0.9)
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("TextureAnalyzer: GLCM computation failed: %s", exc)
            cls._write_neutral(frame)

    @classmethod
    def _write_neutral(cls, frame: AnalysisFrame) -> None:
        frame.add_attribute("texture.glcm.contrast", 0.0, confidence=0.1)
        frame.add_attribute("texture.glcm.homogeneity", 0.5, confidence=0.1)
        frame.add_attribute("texture.glcm.energy", 0.5, confidence=0.1)