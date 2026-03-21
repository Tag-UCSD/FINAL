import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class AnalysisFrame:
    """
    Standard unit of analysis for the v3 pipeline.
    Now extended to support Depth Maps and Semantic Segmentation for higher-order science.
    """
    image_id: int
    original_image: np.ndarray  # RGB, uint8
    
    # Derived data (populated by pipeline)
    gray_image: Optional[np.ndarray] = None
    lab_image: Optional[np.ndarray] = None  # CIELAB for perceptual color
    edges: Optional[np.ndarray] = None
    
    # Future-proofing for Phase 3.2 (Depth)
    depth_map: Optional[np.ndarray] = None 
    
    # Results
    attributes: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Lazy load opencv only when needed
        import cv2
        from skimage import color
        
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        
        if self.edges is None:
            # L2gradient=True provides more accurate edge magnitude for architecture
            self.edges = cv2.Canny(self.gray_image, 50, 150, L2gradient=True)

        if self.lab_image is None:
            # Convert to LAB for scientifically valid color analysis
            # We use skimage because cv2's LAB scaling is non-standard/confusing
            self.lab_image = color.rgb2lab(self.original_image)

    def add_attribute(self, key: str, value: float, confidence: float = 1.0):
        """
        Add a computed attribute to the frame.
        Value should generally be normalized 0.0 - 1.0 where possible.
        """
        self.attributes[key] = float(value)
        self.metadata[key] = {"confidence": confidence}