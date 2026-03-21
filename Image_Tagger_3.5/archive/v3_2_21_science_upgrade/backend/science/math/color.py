"""
Color analysis module for v3 Science Pipeline.

Adapts key ideas from v2.6.3 color extractor into the v3 AnalysisFrame
pattern. Computes a small set of robust, interpretable metrics:
- luminance mean / std
- color temperature proxy
- saturation mean
- hue entropy (color diversity)
- warm vs cool dominance
- composite "color richness"
"""

from __future__ import annotations

import numpy as np
import cv2

from backend.science.core import AnalysisFrame


class ColorAnalyzer:
    """Static color-analysis utilities for an AnalysisFrame."""

    @staticmethod
    def analyze(frame: AnalysisFrame) -> None:
        img = frame.original_image
        if img is None:
            return

        # Normalize to uint8 RGB
        if img.dtype == np.float32 or img.dtype == np.float64:
            image_uint8 = (img * 255).astype(np.uint8)
        else:
            image_uint8 = img

        # Grayscale for luminance
        gray = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2GRAY)
        mean_lum = float(np.mean(gray) / 255.0)
        std_lum = float(np.std(gray) / 255.0)

        # HSV for hue/saturation analysis
        hsv = cv2.cvtColor(image_uint8, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]

        # Color temperature proxy (ported from v2 logic)
        color_temp = ColorAnalyzer._compute_color_temperature(image_uint8)

        # Hue entropy (diversity of colors)
        hue_entropy = ColorAnalyzer._compute_hue_entropy(hue)

        # Saturation mean
        sat_mean = float(np.mean(sat) / 255.0)

        # Warm vs cool dominance
        warm_ratio = ColorAnalyzer._compute_warm_cool_ratio(hsv)

        # Composite color richness: saturation × hue entropy
        color_richness = sat_mean * hue_entropy

        frame.add_attribute("color.luminance_mean", mean_lum)
        frame.add_attribute("color.luminance_std", std_lum)
        frame.add_attribute("color.temperature", color_temp)
        frame.add_attribute("color.saturation_mean", sat_mean)
        frame.add_attribute("color.hue_entropy", hue_entropy)
        frame.add_attribute("color.warm_color_dominance", warm_ratio)
        frame.add_attribute("color.richness", color_richness)

    @staticmethod
    def _compute_color_temperature(image_rgb: np.ndarray) -> float:
        """
        Estimate normalized color temperature based on red/blue ratio.

        Adapted from v2: maps approximate Kelvin range [2000, 10000]K
        into a [0, 1] scalar for downstream use.
        """
        r_mean = float(np.mean(image_rgb[:, :, 0]))
        b_mean = float(np.mean(image_rgb[:, :, 2]))

        if b_mean == 0.0:
            b_mean = 1.0

        rb_ratio = r_mean / b_mean

        # Simplified mapping from v2
        if rb_ratio > 1.2:
            # Warm (2000–4000K)
            temp_k = 2000.0 + (rb_ratio - 1.2) * 2000.0
        elif rb_ratio < 0.8:
            # Cool (6000–10000K)
            temp_k = 10000.0 - (0.8 - rb_ratio) * 4000.0
        else:
            # Neutral (4000–6000K)
            temp_k = 4000.0 + (rb_ratio - 0.8) * 5000.0

        temp_normalized = (temp_k - 2000.0) / 8000.0
        return float(np.clip(temp_normalized, 0.0, 1.0))

    @staticmethod
    def _compute_hue_entropy(hue_channel: np.ndarray) -> float:
        """
        Compute normalized entropy of hue distribution.

        Direct adaptation of v2 _compute_hue_entropy implementation,
        but as a static utility.
        """
        hist, _ = np.histogram(hue_channel.flatten(), bins=180, range=(0, 180))
        hist = hist.astype(np.float64)
        hist = hist / (np.sum(hist) + 1e-7)

        entropy = -np.sum(hist * np.log2(hist + 1e-7))
        max_entropy = np.log2(180.0)
        return float(entropy / max_entropy)

    @staticmethod
    def _compute_warm_cool_ratio(hsv: np.ndarray) -> float:
        """
        Compute ratio of warm to cool colors.

        Warm hues: 0–60 in OpenCV HSV
        Cool hues: 90–150 in OpenCV HSV
        """
        hue = hsv[:, :, 0]

        warm_mask = (hue >= 0) & (hue <= 60)
        cool_mask = (hue >= 90) & (hue <= 150)

        warm_count = float(np.sum(warm_mask))
        cool_count = float(np.sum(cool_mask))
        total = warm_count + cool_count

        if total == 0.0:
            return 0.5  # Neutral

        warm_ratio = warm_count / total
        return float(warm_ratio)