"""
Low-Level Computer Vision Logic.
Consolidates Color, Texture, and Geometry extractors into a single optimized module.
"""
import numpy as np
import cv2
from backend.science.core import AnalysisFrame

class VisionProcessor:
    
    @staticmethod
    def extract_color_features(frame: AnalysisFrame):
        """Extracts Luminance, Temperature, and Saturation."""
        img = frame.original_image
        
        # 1. Mean Luminance
        mean_lum = np.mean(frame.gray_image) / 255.0
        frame.add_attribute('color.luminance', mean_lum)
        
        # 2. Saturation (HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        sat = np.mean(hsv[:, :, 1]) / 255.0
        frame.add_attribute('color.saturation', sat)
        
        # 3. Warm/Cool Ratio
        # Hue: 0-60 (Warm), 90-150 (Cool) in OpenCV
        hue = hsv[:, :, 0]
        warm_pixels = np.sum((hue >= 0) & (hue <= 60))
        cool_pixels = np.sum((hue >= 90) & (hue <= 150))
        total = warm_pixels + cool_pixels + 1e-6
        frame.add_attribute('color.warmth', warm_pixels / total)

    @staticmethod
    def extract_geometry_features(frame: AnalysisFrame):
        """Extracts Complexity, Edges, and Symmetry."""
        gray = frame.gray_image
        h, w = gray.shape
        
        # 1. Edge Density (Complexity Proxy)
        edge_pixels = np.sum(frame.edges > 0)
        density = edge_pixels / (h * w)
        frame.add_attribute('complexity.edge_density', min(density * 5, 1.0)) # Scale 0-1
        
        # 2. Symmetry (Horizontal Flip correlation)
        left = gray[:, :w//2]
        right = gray[:, w//2:]
        # Handle odd widths
        min_w = min(left.shape[1], right.shape[1])
        
        # Flip right side
        right_flipped = np.fliplr(right[:, :min_w])
        left_trimmed = left[:, -min_w:]
        
        # Simple correlation
        corr = np.corrcoef(left_trimmed.flatten(), right_flipped.flatten())[0, 1]
        symmetry_score = max(0, (corr + 1) / 2) # Normalize -1..1 to 0..1
        frame.add_attribute('fluency.symmetry', symmetry_score)

    @staticmethod
    def run_all(frame: AnalysisFrame):
        VisionProcessor.extract_color_features(frame)
        VisionProcessor.extract_geometry_features(frame)