"""Spatial structure, depth maps, and clutter proxies for depth-like properties.

This module implements 2D edge/clutter heuristics as proxies for
depth-related concepts (openness, refuge, isovist area), and optionally
integrates a monocular depth estimation model via ONNX Runtime when
configured.

The goal is to make the depth layer *scientifically honest*:
- If a depth model is available, we compute a normalized depth map and
  expose simple summary statistics as BN variables.
- If not, we fall back to the existing edge-based heuristics so the
  rest of the pipeline continues to function.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np

from backend.science.core import AnalysisFrame

logger = logging.getLogger(__name__)

try:  # Optional dependency; we degrade gracefully if missing.
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover - onnxruntime not installed
    ort = None  # type: ignore


class DepthAnalyzer:
    """Analyze spatial structure using edges and optional monocular depth.

    This class maintains backward-compatible 2D edge/clutter proxies while
    optionally computing a true depth map when an ONNX model is available.

    Configuration
    -------------
    If you wish to enable monocular depth:

    1. Install onnxruntime in the backend environment.
    2. Download a suitable depth model (e.g. DepthAnything / MiDaS) in ONNX
       format.
    3. Set the environment variable ``DEPTH_ANYTHING_ONNX_PATH`` to the
       model path inside the container.

    When misconfigured (missing library or model), the analyzer logs a
    warning and silently falls back to the legacy heuristics.
    """

    _onnx_session: Optional["ort.InferenceSession"] = None

    @classmethod
    def _get_onnx_session(cls) -> Optional["ort.InferenceSession"]:
        """Lazily initialise the ONNX Runtime session if possible."""
        if ort is None:
            return None
        if cls._onnx_session is not None:
            return cls._onnx_session

        model_path = os.getenv("DEPTH_ANYTHING_ONNX_PATH")
        if not model_path or not os.path.exists(model_path):
            return None

        try:
            cls._onnx_session = ort.InferenceSession(  # type: ignore[attr-defined]
                model_path,
                providers=["CPUExecutionProvider"],
            )
            logger.info("DepthAnalyzer: loaded depth ONNX model from %s", model_path)
        except Exception:  # pragma: no cover - runtime environment dependent
            logger.warning(
                "DepthAnalyzer: failed to load depth ONNX model from %s; "
                "falling back to edge-based heuristics.",
                model_path,
                exc_info=True,
            )
            cls._onnx_session = None

        return cls._onnx_session

    @classmethod
    def _compute_depth_map(cls, frame: AnalysisFrame) -> Optional[np.ndarray]:
        """Compute a normalized depth map for the frame, if configured.

        Returns a H×W float32 array in [0, 1] when successful, otherwise
        ``None``. Any errors cause a clean fallback to the legacy
        edge-based heuristics.
        """
        session = cls._get_onnx_session()
        if session is None:
            return None

        if frame.original_image is None:
            return None

        try:
            import cv2  # type: ignore
        except Exception:  # pragma: no cover - cv2 not available
            logger.warning(
                "DepthAnalyzer: OpenCV (cv2) not available; "
                "cannot run ONNX depth model. Falling back to heuristics."
            )
            return None

        img = frame.original_image
        if img.ndim == 2:  # grayscale → fake 3-channel
            img = np.stack([img, img, img], axis=-1)

        # Many monocular models expect a square input; 384×384 is common.
        target_size = (384, 384)
        resized = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        inp = resized.astype(np.float32) / 255.0
        inp = np.transpose(inp, (2, 0, 1))[None, ...]  # NCHW

        try:
            input_name = session.get_inputs()[0].name
            outputs = session.run(None, {input_name: inp})
        except Exception:  # pragma: no cover - runtime specific
            logger.warning(
                "DepthAnalyzer: ONNX inference failed; "
                "falling back to edge-based heuristics.",
                exc_info=True,
            )
            return None

        depth = np.squeeze(outputs[0])
        if depth.ndim == 3:
            depth = depth[0]

        h, w = frame.original_image.shape[:2]
        depth_resized = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)

        d_min = float(depth_resized.min())
        d_max = float(depth_resized.max())
        if d_max > d_min:
            depth_norm = (depth_resized - d_min) / (d_max - d_min)
        else:
            depth_norm = np.zeros_like(depth_resized, dtype=np.float32)

        return depth_norm.astype(np.float32)

    @staticmethod
    def _summarise_depth(depth: np.ndarray) -> tuple[float, float]:
        """Return (mean_depth, depth_contrast) in [0, 1]."""
        if depth.size == 0:
            return 0.0, 0.0

        mean_depth = float(depth.mean())
        std_depth = float(depth.std())

        # Clamp to [0, 1] for BN friendliness
        norm_mean = float(np.clip(mean_depth, 0.0, 1.0))
        # Contrast emphasises mid-range variance but avoids extreme outliers
        contrast = float(np.clip(std_depth * 0.5, 0.0, 1.0))
        return norm_mean, contrast

    def analyze(self, frame: AnalysisFrame) -> None:
        """Compute depth-related attributes for a single frame.

        - Always produces the legacy edge-based clutter/openness/refuge
          proxies used by existing BN scripts.
        - Optionally computes a depth map and summary statistics when a
          monocular depth model is configured.
        """
        # Optional: true depth map via ONNX
        depth_map = self._compute_depth_map(frame)
        if depth_map is not None:
            frame.depth_map = depth_map
            mean_depth, depth_contrast = self._summarise_depth(depth_map)
            frame.add_attribute("spatial.depth_mean", mean_depth)
            frame.add_attribute("spatial.depth_contrast", depth_contrast)

        # Always compute legacy 2D edge proxies so existing BN exports
        # remain stable, even when no depth model is available.
        if frame.edges is None:
            # AnalysisFrame.precompute_basics() should normally ensure edges,
            # but if it was skipped we avoid crashing the pipeline.
            logger.warning(
                "DepthAnalyzer: frame.edges is None; skipping clutter/openness "
                "computation for this frame."
            )
            return

        # 1. Visual Clutter (Edge Variance)
        clutter_score = DepthAnalyzer.calculate_clutter_proxy(frame.edges)
        frame.add_attribute("spatial.visual_clutter", clutter_score)

        # 2. Central Openness (Prospect proxy)
        openness = DepthAnalyzer.calculate_central_openness(frame.edges)
        frame.add_attribute("spatial.central_openness", openness)

        # 3. Refuge Quality (Bayesian Node)
        refuge = DepthAnalyzer.calculate_refuge_quality(frame)
        frame.add_attribute("spatial.refuge_quality", refuge)

        # 4. Isovist Area (Affordance)
        frame.add_attribute("affordance.isovist_area", openness * 0.8)

    @staticmethod
    def calculate_clutter_proxy(edges: np.ndarray) -> float:
        """Heuristic clutter score based on edge density variance.

        High variance in edge density across a grid implies disordered
        clutter; we normalise the result to [0, 1].
        """
        h, w = edges.shape
        grid_h, grid_w = max(h // 8, 1), max(w // 8, 1)
        variances = []
        for y in range(0, h, grid_h):
            for x in range(0, w, grid_w):
                cell = edges[y : y + grid_h, x : x + grid_w]
                if cell.size > 0:
                    density = np.count_nonzero(cell) / float(cell.size)
                    variances.append(density)
        if not variances:
            return 0.0
        return float(min(np.std(variances) * 5.0, 1.0))

    @staticmethod
    def calculate_central_openness(edges: np.ndarray) -> float:
        """Prospect proxy based on edge density in the central window."""
        h, w = edges.shape
        cy, cx = h // 2, w // 2
        dy, dx = max(h // 6, 1), max(w // 6, 1)
        center_crop = edges[cy - dy : cy + dy, cx - dx : cx + dx]
        if center_crop.size == 0:
            return 0.0
        edge_density = np.count_nonzero(center_crop) / float(center_crop.size)
        return float(1.0 - min(edge_density * 5.0, 1.0))

    @staticmethod
    def calculate_refuge_quality(frame: AnalysisFrame) -> float:
        """Estimate 'Refuge' potential from near-floor occlusion / shelter.

        This metric now prefers depth-map-aware reasoning when a monocular
        depth model is configured, and falls back to the legacy edge-based
        heuristic otherwise.

        Intuition
        ---------
        - In a depth map, *near* pixels (small depth) near the bottom of the
          frame are interpreted as potential occluding / sheltering elements
          (sofas, counters, railings, low walls).
        - We summarise this using the average *nearness* (1 - normalised depth)
          over the bottom band of the frame.
        - If no depth map is available, we retain the previous Canny-based
          proxy so the rest of the pipeline continues to function.
        """
        # Preferred path: use the depth map if available.
        depth = getattr(frame, "depth_map", None)
        if depth is not None:
            arr = np.asarray(depth, dtype="float32")
            if arr.ndim == 3:
                # Handle HxWx1 or HxWxC by collapsing channels.
                arr = arr[..., 0]
            if arr.ndim != 2 or arr.size == 0:
                return 0.0

            h, _w = arr.shape
            if h < 4:
                return 0.0

            # Bottom 30% of the frame as a near-floor / near-viewing band.
            start_row = int(h * 0.7)
            band = arr[start_row:, :]
            if band.size == 0:
                return 0.0

            # Robust per-band normalisation to [0, 1]
            band_min = float(np.nanmin(band))
            band_max = float(np.nanmax(band))
            if not np.isfinite(band_min) or not np.isfinite(band_max):
                return 0.0
            if band_max > band_min:
                norm = (band - band_min) / (band_max - band_min)
            else:
                norm = np.zeros_like(band, dtype="float32")

            # Higher nearness = stronger sense of nearby refuge.
            nearness = 1.0 - norm
            score = float(np.clip(np.nanmean(nearness), 0.0, 1.0))
            return score

        # Fallback: legacy edge-density heuristic.
        edges = frame.edges
        if edges is None:
            return 0.0

        h, _w = edges.shape
        if h < 4:
            return 0.0

        bottom_crop = edges[int(h * 0.8) :, :]
        if bottom_crop.size == 0:
            return 0.0

        occ_density = np.count_nonzero(bottom_crop) / float(bottom_crop.size)
        return float(np.clip(occ_density * 2.0, 0.0, 1.0))
