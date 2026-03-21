import numpy as np
from skimage.feature import graycomatrix, graycoprops
from backend.science.core import AnalysisFrame

class TextureAnalyzer:
    """
    GLCM Texture Analysis.
    Updated to expose multi-scale texture detection (near vs far).
    """
    
    @staticmethod
    def analyze(frame: AnalysisFrame):
        gray = frame.gray_image
        # Downsample for performance
        h, w = gray.shape
        if h > 512:
            scale = 512 / h
            gray = cv2.resize(gray, (0,0), fx=scale, fy=scale)

        # Quantize to 64 levels
        gray = (gray // 4).astype(np.uint8)

        # Analyze at two distances: 1 (Micro-texture) and 5 (Macro-structure)
        distances = [1, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        glcm = graycomatrix(gray, distances=distances, angles=angles, 
                            levels=64, symmetric=True, normed=True)
        
        # Extract features and average across angles
        # property array shape: (num_distances, num_angles)
        contrast = graycoprops(glcm, 'contrast')
        energy = graycoprops(glcm, 'energy')
        homogeneity = graycoprops(glcm, 'homogeneity')

        # Micro-Texture (Distance 1)
        frame.add_attribute("texture.micro.contrast", np.mean(contrast[0]))
        frame.add_attribute("texture.micro.homogeneity", np.mean(homogeneity[0]))

        # Macro-Texture (Distance 5)
        frame.add_attribute("texture.macro.contrast", np.mean(contrast[1]))
        frame.add_attribute("texture.macro.homogeneity", np.mean(homogeneity[1]))