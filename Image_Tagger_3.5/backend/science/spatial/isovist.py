import numpy as np
import cv2
from backend.science.core import AnalysisFrame

class IsovistAnalyzer:
    """
    Implements 2D Isovist analysis on the floor plan (or segmentation mask).
    Calculates 'Visual Area' from a vantage point.
    """

    @staticmethod
    def compute_2d_isovist(frame: AnalysisFrame, floor_mask: np.ndarray = None):
        """
        Simulates a 360-degree raycast from the image center (or floor center).
        If no floor_mask provided, uses basic edge avoidance.
        """
        h, w, _ = frame.original_image.shape
        center = (w // 2, h // 2)
        
        # Create an obstacle map from edges (simplistic proxy for walls)
        obstacles = frame.edges > 0
        
        # Raycasting (Simplified 36-ray sweep)
        # In production, use a proper visibility polygon algorithm
        visible_distance_sum = 0
        rays = 36
        max_dist = np.sqrt(h**2 + w**2) / 2
        
        for i in range(rays):
            angle = (i * 360 / rays) * (np.pi / 180)
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            dist = 0
            for r in range(1, int(max_dist)):
                cx, cy = int(center[0] + dx * r), int(center[1] + dy * r)
                
                # Bounds check
                if cx < 0 or cx >= w or cy < 0 or cy >= h:
                    break
                    
                # Obstacle check (Hit a wall/edge)
                if obstacles[cy, cx]:
                    break
                
                dist = r
            
            visible_distance_sum += dist
            
        # Normalize "Openness" based on average ray length vs image size
        avg_dist = visible_distance_sum / rays
        openness_score = min(avg_dist / (min(h,w)/2), 1.0)
        
        frame.add_attribute("spatial.isovist_openness", openness_score)