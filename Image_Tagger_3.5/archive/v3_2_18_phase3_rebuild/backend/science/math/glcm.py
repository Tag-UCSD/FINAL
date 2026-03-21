import numpy as np
from skimage.feature import graycomatrix, graycoprops
from backend.science.core import AnalysisFrame

class TextureAnalyzer:
    """
    Uses Gray-Level Co-occurrence Matrices (GLCM) to measure texture quality.
    """
    
    @staticmethod
    def analyze(frame: AnalysisFrame):
        # GLCM requires uint8 image
        gray = frame.gray_image
        
        # Compute GLCM
        # distances=[1], angles=[0, 45, 90, 135]
        glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                            levels=256, symmetric=True, normed=True)
        
        # Extract Properties
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
        energy = graycoprops(glcm, 'energy')[0, 0]
        correlation = graycoprops(glcm, 'correlation')[0, 0]
        
        # Normalize Contrast (heuristic)
        contrast_norm = min(contrast / 1000.0, 1.0)
        
        frame.add_attribute("texture.glcm.contrast", contrast_norm)
        frame.add_attribute("texture.glcm.homogeneity", homogeneity)
        frame.add_attribute("texture.glcm.energy", energy)