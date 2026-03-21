"""Spatial structure analyzer (placeholder for depth-based metrics)."""

from __future__ import annotations

import numpy as np
from backend.science.core import AnalysisFrame


class DepthAnalyzer:
    """Compute simple edge-based spatial structure proxies.

    This is a bridge towards a future depth-based module. It provides:
      - spatial.edge_clutter_sd: SD of edge density across a coarse grid
      - spatial.central_openness: 1 - edge density in central crop
      - spatial.vertical_balance: edge density difference top vs bottom
    """

    def __init__(self, grid_size: int = 8, central_fraction: float = 0.4):
        self.grid_size = max(2, grid_size)
        self.central_fraction = float(np.clip(central_fraction, 0.1, 0.9))

    def analyze(self, frame: AnalysisFrame) -> None:
        edges = frame.ensure_edges()
        h, w = edges.shape
        if h < 2 or w < 2:
            return

        # 1. Edge clutter SD across grid
        gh = min(self.grid_size, h)
        gw = min(self.grid_size, w)
        cell_h = max(1, h // gh)
        cell_w = max(1, w // gw)

        densities = []
        for gy in range(gh):
            y0 = gy * cell_h
            y1 = min(h, (gy + 1) * cell_h)
            if y0 >= h:
                break
            for gx in range(gw):
                x0 = gx * cell_w
                x1 = min(w, (gx + 1) * cell_w)
                if x0 >= w:
                    break
                cell = edges[y0:y1, x0:x1]
                if cell.size == 0:
                    continue
                dens = float((cell > 0).sum()) / float(cell.size)
                densities.append(dens)

        if len(densities) > 0:
            clutter_sd = float(np.std(densities))
            clutter_norm = float(np.clip(clutter_sd / 0.5, 0.0, 1.0))
        else:
            clutter_norm = 0.0

        # 2. Central openness (fewer edges in centre => higher openness)
        cf = self.central_fraction
        cy0 = int((h * (1.0 - cf)) / 2.0)
        cy1 = int((h * (1.0 + cf)) / 2.0)
        cx0 = int((w * (1.0 - cf)) / 2.0)
        cx1 = int((w * (1.0 + cf)) / 2.0)
        centre = edges[cy0:cy1, cx0:cx1]
        if centre.size > 0:
            centre_density = float((centre > 0).sum()) / float(centre.size)
        else:
            centre_density = 0.0
        central_openness = float(np.clip(1.0 - centre_density, 0.0, 1.0))

        # 3. Vertical balance (edges top vs bottom)
        half = h // 2
        top = edges[:half, :]
        bottom = edges[half:, :]
        if top.size > 0:
            top_d = float((top > 0).sum()) / float(top.size)
        else:
            top_d = 0.0
        if bottom.size > 0:
            bot_d = float((bottom > 0).sum()) / float(bottom.size)
        else:
            bot_d = 0.0
        vert_balance = float(np.clip((bot_d - top_d + 1.0) / 2.0, 0.0, 1.0))

        frame.set_attributes(
            {
                "spatial.edge_clutter_sd": clutter_norm,
                "spatial.central_openness": central_openness,
                "spatial.vertical_balance": vert_balance,
            }
        )