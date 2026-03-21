import numpy as np
import cv2
from scipy.stats import entropy
from skimage.feature import graycomatrix
from backend.science.core import AnalysisFrame

class ComplexityAnalyzer:
    """
    Quantifies 'Visual Complexity' using both Information Theory (Entropy)
    and Structural Analysis (Spatial Entropy).
    """

    @staticmethod
    def calculate_shannon_entropy(image_gray: np.ndarray) -> float:
        """
        Global histogram entropy. Measures 'amount of information' but 
        ignores spatial arrangement (snow vs checkerboard).
        """
        hist = cv2.calcHist([image_gray], [0], None, [256], [0, 256])
        hist = hist.ravel() / hist.sum()
        return entropy(hist, base=2)

    @staticmethod
    def calculate_spatial_entropy(image_gray: np.ndarray) -> float:
        """
        Measures entropy of the GLCM. This captures 'spatial disorder'.
        A checkerboard has Low Spatial Entropy (high order).
        White noise has High Spatial Entropy (low order).
        """
        # Downscale for speed if needed, GLCM is expensive
        h, w = image_gray.shape
        if h > 512:
            scale = 512 / h
            small = cv2.resize(image_gray, (0,0), fx=scale, fy=scale)
        else:
            small = image_gray
            
        # Quantize to 32 levels to stabilize GLCM
        small_quant = (small // 8).astype(np.uint8)
        
        glcm = graycomatrix(small_quant, distances=[1], angles=[0, np.pi/4, np.pi/2], 
                            levels=32, symmetric=True, normed=True)
        
        # Compute entropy of the non-zero GLCM elements
        glcm_flat = glcm.flatten()
        glcm_flat = glcm_flat[glcm_flat > 0]
        spatial_ent = -np.sum(glcm_flat * np.log2(glcm_flat))
        
        # Normalize (Max entropy for 32x32 matrix is log2(32*32) = 10)
        return min(spatial_ent / 10.0, 1.0)

    @staticmethod
    def analyze(frame: AnalysisFrame):
        # 1. Global Entropy (Abundance of gray levels)
        ent = ComplexityAnalyzer.calculate_shannon_entropy(frame.gray_image)
        # Normalize roughly (8 bits = max 8)
        frame.add_attribute("complexity.shannon_entropy", min(ent / 8.0, 1.0))
        
        # 2. Spatial Entropy (Disorder of texture)
        spatial = ComplexityAnalyzer.calculate_spatial_entropy(frame.gray_image)
        frame.add_attribute("complexity.spatial_entropy", spatial)
        
        # 3. Edge Density (Clutter Proxy)
        # Simple ratio of Canny pixels to total area
        total_pixels = frame.edges.size
        edge_pixels = np.count_nonzero(frame.edges)
        frame.add_attribute("complexity.edge_density", edge_pixels / total_pixels)