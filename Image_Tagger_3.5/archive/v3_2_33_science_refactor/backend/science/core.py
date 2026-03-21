import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Any, Optional

try:
    import cv2
except ImportError:  # pragma: no cover - dependency checked at runtime
    cv2 = None

try:
    from skimage.color import rgb2lab
except ImportError:  # pragma: no cover
    rgb2lab = None


@dataclass
class AnalysisFrame:
    """Central data structure for the v3.3 science pipeline.

    This wraps a single RGB image and caches common derived
    representations used by the analyzers (gray, LAB, edges, depth).
    Additional attributes and metadata are accumulated as the pipeline runs.
    """

    image_id: int
    original_image: np.ndarray  # RGB uint8, shape (H, W, 3)

    gray_image: Optional[np.ndarray] = None
    lab_image: Optional[np.ndarray] = None
    edges: Optional[np.ndarray] = None

    depth_map: Optional[np.ndarray] = None
    semantic_segmentation: Optional[np.ndarray] = None

    attributes: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def ensure_gray(self) -> np.ndarray:
        """Return a grayscale version of the image, computing if necessary."""
        if self.gray_image is None:
            if cv2 is None:
                # simple luminance formula as a fallback
                rgb = self.original_image.astype(np.float32) / 255.0
                r = rgb[:, :, 0]
                g = rgb[:, :, 1]
                b = rgb[:, :, 2]
                self.gray_image = (
                    0.2126 * r
                    + 0.7152 * g
                    + 0.0722 * b
                ).astype(np.float32)
            else:
                self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_RGB2GRAY)
        return self.gray_image

    def ensure_lab(self) -> np.ndarray:
        """Return CIELAB representation, computing if necessary.

        We prefer skimage.color.rgb2lab; if unavailable we approximate
        using a simple linear mapping from RGB to L and zero a/b.
        """
        if self.lab_image is None:
            rgb = self.original_image.astype(np.float32) / 255.0
            if rgb2lab is not None:
                lab = rgb2lab(rgb)
                self.lab_image = lab.astype(np.float32)
            else:
                # Fallback: compute L* as luminance; set a/b to zero.
                gray = self.ensure_gray().astype(np.float32)
                L = gray / 255.0 * 100.0  # pseudo-L*
                a = np.zeros_like(L)
                b = np.zeros_like(L)
                self.lab_image = np.stack([L, a, b], axis=-1)
        return self.lab_image

    def ensure_edges(self) -> np.ndarray:
        """Return an edge map (uint8 0/255), computing if necessary."""
        if self.edges is None:
            gray = self.ensure_gray()
            if cv2 is None:
                # basic gradient-magnitude threshold as fallback
                gy, gx = np.gradient(gray.astype(np.float32))
                mag = np.hypot(gx, gy)
                thresh = float(np.percentile(mag, 75))
                self.edges = (mag > thresh).astype(np.uint8) * 255
            else:
                self.edges = cv2.Canny(gray, threshold1=100, threshold2=200)
        return self.edges

    def set_attribute(self, key: str, value: float) -> None:
        """Store a single scalar attribute, clamped to [0, 1] where sensible."""
        if isinstance(value, (int, float)):
            v = float(value)
            if not np.isfinite(v):
                v = 0.0
            if v < 0.0:
                v = 0.0
            if v > 1.0:
                v = 1.0
            self.attributes[key] = v

    def set_attributes(self, values: Dict[str, float]) -> None:
        for k, v in values.items():
            self.set_attribute(k, v)