import numpy as np
import cv2
from backend.science.core import AnalysisFrame

class FractalAnalyzer:
    """
    Implements Box Counting Method for Fractal Dimension (D).
    Standard architecture metric for 'visual richness'.
    """

    @staticmethod
    def analyze(frame: AnalysisFrame):
        # Use the pre-computed edges from the AnalysisFrame
        # This ensures we measure the D of the *structure*, not the noise
        d_score = FractalAnalyzer.box_counting(frame.edges)
        
        # Fractal D usually ranges 1.0 (Line) to 2.0 (Plane).
        # We normalize 1.0 -> 2.0 to 0.0 -> 1.0 for the DB
        norm_d = max(0.0, min((d_score - 1.0), 1.0))
        frame.add_attribute("fractal.D", norm_d)

    @staticmethod
    def box_counting(Z: np.ndarray) -> float:
        """
        Minkowski-Bouligand dimension.
        Z: Binary array (edges).
        """
        if np.sum(Z) == 0:
            return 0.0

        # Only check up to min dimension / 2
        p = min(Z.shape)
        n = int(np.floor(np.log(p)/np.log(2)))
        sizes = 2**np.arange(n, 1, -1)
        
        counts = []
        for size in sizes:
            # Fast box counting using add.reduceat
            count = FractalAnalyzer._fast_box_count(Z, size)
            counts.append(count)

        # Linear Regression on log-log scale
        # Fit: log(N) = D * log(1/s) + c
        if len(counts) < 2: 
            return 0.0
            
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0] # The slope is -D

    @staticmethod
    def _fast_box_count(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k*k))[0])