"""
Depth provider plugin interface.

This module defines a minimal protocol for plugging in a monocular depth
estimator without tying the science pipeline to a particular model.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np
from backend.science.core import AnalysisFrame
from backend.science.contracts import fail


class DepthProvider(Protocol):
    name: str

    def infer_depth(self, image_rgb: np.ndarray) -> np.ndarray:
        raise NotImplementedError("DepthProvider is a protocol; implement infer_depth in a plugin.")


class DepthPluginAnalyzer:
    """
    Optional analyzer that calls a configured DepthProvider and stores
    the result in the AnalysisFrame. By default this is a no-op stub
    until a provider is configured.
    """

    name = "depth_plugin"
    tier = "L2"
    requires = ["original_image"]
    provides = ["depth_map"]

    # TODO: wire in a real provider via configuration.
    provider: DepthProvider | None = None

    @classmethod
    def analyze(cls, frame: AnalysisFrame) -> None:
        if cls.provider is None:
            fail(frame, cls.name, "no depth provider configured")
            return

        depth = cls.provider.infer_depth(frame.original_image)
        frame.depth_map = depth
