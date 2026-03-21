import numpy as np
import cv2
from backend.science.core import AnalysisFrame

class FractalAnalyzer:
    """
    Implements the 'Box Counting' method to estimate Fractal Dimension (D).
    Can be applied globally or to specific segmentation masks (regions).
    """

    @staticmethod
    def fractal_dimension(image_array: np.ndarray, threshold=0.9) -> float:
        """
        Calculates Minkowski–Bouligand dimension (Box-counting dimension).
        """
        # 1. Binarize (Edges or Threshold)
        # If image is not binary, run Canny edge detection first
        if len(image_array.shape) > 2:
            gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            Z = edges > 0
        else:
            Z = image_array > threshold

        # 2. Minimal checking
        if np.sum(Z) == 0:
            return 0.0

        # 3. Box Counting
        p = min(Z.shape)
        n = 2**np.floor(np.log(p)/np.log(2))
        n = int(np.log(n)/np.log(2))
        sizes = 2**np.arange(n, 1, -1)
        counts = []

        for size in sizes:
            counts.append(FractalAnalyzer._box_count(Z, size))

        # 4. Linear Regression to find D (Slope of log-log plot)
        coeffs = np.polyfit(np.log(sizes), np.log(counts), 1)
        return -coeffs[0]

    @staticmethod
    def _box_count(Z, k):
        S = np.add.reduceat(
            np.add.reduceat(Z, np.arange(0, Z.shape[0], k), axis=0),
                               np.arange(0, Z.shape[1], k), axis=1)
        return len(np.where((S > 0) & (S < k*k))[0])

    @staticmethod
    def analyze_regions(frame: AnalysisFrame, segmentation_masks: dict):
        """
        Calculates D for specific architectural elements (e.g., Walls vs Floors).
        """
        for label, mask in segmentation_masks.items():
            # Mask the edge map with the region
            masked_edges = np.logical_and(frame.edges > 0, mask > 0)
            d_score = FractalAnalyzer.fractal_dimension(masked_edges)
            
            # Store as a localized attribute
            frame.add_attribute(f"fractal.D.{label}", d_score)