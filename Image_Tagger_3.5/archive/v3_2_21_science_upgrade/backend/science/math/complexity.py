import numpy as np
import cv2
from scipy.stats import entropy
from backend.science.core import AnalysisFrame

class ComplexityAnalyzer:
    """
    Quantifies 'Visual Complexity' using Shannon Entropy and Edge Density.
    High Complexity + High Order = 'Organized Complexity' (Goldilocks Zone).
    """

    @staticmethod
    def calculate_shannon_entropy(image: np.ndarray) -> float:
        """
        Measures the 'information content' of the image texture.
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
            
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        
        # Compute entropy
        return entropy(hist, base=2)

    @staticmethod
    def calculate_edge_density(frame: AnalysisFrame) -> float:
        """
        Ratio of edge pixels to total pixels. A proxy for visual clutter.
        """
        total_pixels = frame.edges.size
        edge_pixels = np.count_nonzero(frame.edges)
        return edge_pixels / total_pixels

    @staticmethod
    def analyze(frame: AnalysisFrame):
        # 1. Raw Complexity
        ent = ComplexityAnalyzer.calculate_shannon_entropy(frame.original_image)
        frame.add_attribute("complexity.entropy", ent)
        
        # 2. Structural Density
        dens = ComplexityAnalyzer.calculate_edge_density(frame)
        frame.add_attribute("complexity.edge_density", dens)
        
        # 3. 'Organized Complexity' Proxy
        # If Entropy is High but Edge Density is Moderate -> Organized
        # If Entropy High and Edge Density High -> Chaotic
        ratio = ent / (dens + 0.01)
        frame.add_attribute("complexity.organization_ratio", ratio)