import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple

@dataclass
class AnalysisFrame:
    """
    Standard unit of analysis for the v3 pipeline.
    Replaces the old 'ImageData' and 'SegmentationData' sprawl.
    """
    image_id: int
    original_image: np.ndarray  # RGB
    
    # Derived data (populated by pipeline)
    gray_image: np.ndarray = None
    edges: np.ndarray = None
    
    # Results
    attributes: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        # Lazy load opencv only when needed
        import cv2
        if self.gray_image is None:
            self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        if self.edges is None:
            self.edges = cv2.Canny(self.gray_image, 50, 150)

    def add_attribute(self, key: str, value: float, confidence: float = 1.0):
        self.attributes[key] = float(value)
        self.metadata[key] = {"confidence": confidence}