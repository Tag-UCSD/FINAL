from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional, Union
import hashlib

import numpy as np

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - cv2 may not be available in tiny CI images
    cv2 = None  # type: ignore

try:
    import requests
except Exception:
    requests = None  # type: ignore

from fastapi import APIRouter, Depends, HTTPException, Response, status
from backend.science import pipeline as science_pipeline
from backend.science.core import AnalysisFrame
from backend.science.spatial.depth import DepthAnalyzer
from backend.science.vision.segmentation import SegmentationAnalyzer
from backend.science.vision.room_detection import RoomDetectionAnalyzer, COARSE_CATEGORIES
from backend.science.vision.clip_material import MaterialIdentificationPipeline
from backend.science.vision.materials import GeminiMaterialAnalyzer
from backend.science.context.affordance import (
    AffordanceAnalyzer, AFFORDANCE_IDS, AFFORDANCE_NAMES,
)
from sqlalchemy.orm import Session

from backend.database.core import get_db
from backend.models.assets import Image  # type: ignore
from backend.services.auth import CurrentUser, require_tagger

router = APIRouter(prefix="/v1/debug", tags=["Debug / Science"])


_MATERIALS2_PIPELINE: Optional[MaterialIdentificationPipeline] = None


def _get_materials2_pipeline() -> MaterialIdentificationPipeline:
    """
    Return a lazily-instantiated singleton MaterialIdentificationPipeline instance.

    This avoids repeatedly loading heavy OneFormer + SigLIP2 weights for each
    /materials2 debug request.
    """
    global _MATERIALS2_PIPELINE
    if _MATERIALS2_PIPELINE is None:
        _MATERIALS2_PIPELINE = MaterialIdentificationPipeline.from_pretrained()
    return _MATERIALS2_PIPELINE


def _is_url(path: str) -> bool:
    """Check if the path is a URL."""
    return path.startswith("http://") or path.startswith("https://")


def _load_image_from_url_or_path(storage_path: str) -> np.ndarray:
    """Load an image from either a URL or a local file path.
    
    Returns the image as a BGR numpy array (OpenCV format).
    Raises HTTPException if the image cannot be loaded.
    """
    if cv2 is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="cv2 (OpenCV) is not available.",
        )

    if _is_url(storage_path):
        # Download image from URL
        if requests is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="requests library is not available; cannot fetch remote images.",
            )
        try:
            response = requests.get(storage_path, timeout=10)
            response.raise_for_status()
            img_array = np.frombuffer(response.content, dtype=np.uint8)
            img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if img_bgr is None:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Could not decode image from URL: {storage_path}",
                )
            return img_bgr
        except requests.RequestException as e:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to download image from URL: {storage_path} - {str(e)}",
            )
    else:
        # Load from local file
        path = _resolve_image_path(storage_path)
        if not path.is_file():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image file not found on disk: {path}",
            )
        img_bgr = cv2.imread(str(path))
        if img_bgr is None:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Could not read image from storage: {path}",
            )
        return img_bgr


def _get_cache_key(storage_path: str) -> str:
    """Generate a cache key from a storage path (works for both URLs and local paths)."""
    if _is_url(storage_path):
        # Use hash of URL for cache key to avoid filesystem issues
        return hashlib.md5(storage_path.encode()).hexdigest()
    else:
        return Path(storage_path).stem


def _resolve_image_path(storage_path: str) -> Path:
    """Resolve the on-disk path for a stored image.

    The storage_path column is expected to contain either an absolute
    path or a path relative to the working directory / IMAGE_STORAGE_ROOT.
    We first try the path as-is; if it does not exist and is relative,
    we fall back to IMAGE_STORAGE_ROOT + storage_path.
    """
    raw = Path(storage_path)
    if raw.is_file():
        return raw

    # Try prefixing with IMAGE_STORAGE_ROOT if provided
    root = os.getenv("IMAGE_STORAGE_ROOT")
    if root:
        candidate = Path(root) / storage_path
        if candidate.is_file():
            return candidate

    return raw  # Best-effort; caller will handle missing file

def _compute_edge_map_bytes(storage_path: str, t1: int = 50, t2: int = 150, l2: bool = True) -> bytes:
    """Compute a Canny edge map PNG for the given image.

    This mirrors the logic in backend.science.core.AnalysisFrame.compute_derived,
    but is implemented locally to keep the debug endpoint self-contained.

    To keep things efficient in classroom settings, we maintain a tiny on-disk
    cache keyed by (image, thresholds, L2 flag). If a matching PNG already
    exists, we serve it directly instead of recomputing.
    
    Supports both local file paths and remote URLs.
    """
    if cv2 is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="cv2 (OpenCV) is not available; cannot compute edge maps.",
        )

    # Compute cache path
    cache_root = os.getenv("IMAGE_DEBUG_CACHE_ROOT") or os.path.join("backend", "data", "debug_edges")
    cache_root_path = Path(cache_root)
    cache_root_path.mkdir(parents=True, exist_ok=True)

    cache_key = _get_cache_key(storage_path)
    cache_name = f"{cache_key}_edges_{t1}_{t2}_{1 if l2 else 0}.png"
    cache_path = cache_root_path / cache_name

    if cache_path.is_file():
        try:
            return cache_path.read_bytes()
        except Exception:
            # Fall through to recomputation on any read error
            pass

    # Load image from URL or local path
    img_bgr = _load_image_from_url_or_path(storage_path)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # Allow experimentation with thresholds and the L2gradient flag
    edges = cv2.Canny(gray, t1, t2, L2gradient=l2)

    ok, buf = cv2.imencode(".png", edges)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode edge map as PNG.",
        )

    data = buf.tobytes()
    try:
        cache_path.write_bytes(data)
    except Exception:
        # Cache write failure should not break the endpoint
        pass
    return data



def _compute_complexity_heatmap_bytes(
    storage_path: str, 
    patch_size: int = 64, 
    stride: int = 32,
    canny_low: int = 50,
    canny_high: int = 150,
) -> bytes:
    """Compute a regionalized complexity heatmap PNG for the given image.

    This implements the edge-density approach from complexity_regions_demo.py:
    For each patch in a sliding window, compute:
        complexity_score = edge_pixels / total_pixels
    
    The result is a heatmap overlaid on the original image, where:
    - Red/Yellow = High complexity (many edges)
    - Dark Red/Black = Low complexity (few edges)
    
    Supports both local file paths and remote URLs.
    """
    if cv2 is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="cv2 (OpenCV) is not available; cannot compute complexity heatmaps.",
        )

    # Compute cache path
    cache_root = os.getenv("IMAGE_COMPLEXITY_CACHE_ROOT") or os.path.join("backend", "data", "debug_complexity")
    cache_root_path = Path(cache_root)
    cache_root_path.mkdir(parents=True, exist_ok=True)

    cache_key = _get_cache_key(storage_path)
    cache_name = f"{cache_key}_complexity_{patch_size}_{stride}_{canny_low}_{canny_high}.png"
    cache_path = cache_root_path / cache_name

    if cache_path.is_file():
        try:
            return cache_path.read_bytes()
        except Exception:
            pass

    # Load image from URL or local path
    img_bgr = _load_image_from_url_or_path(storage_path)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Compute complexity for each patch using sliding window
    out_h = max(1, (h - patch_size) // stride + 1)
    out_w = max(1, (w - patch_size) // stride + 1)
    complexity_map = np.zeros((out_h, out_w), dtype=np.float32)

    for i in range(out_h):
        for j in range(out_w):
            y_start = i * stride
            x_start = j * stride
            patch = gray[y_start:y_start+patch_size, x_start:x_start+patch_size]
            
            # Apply Canny edge detector
            edges = cv2.Canny(patch, canny_low, canny_high)
            
            # Compute edge density = edge_pixels / total_pixels
            edge_pixels = np.count_nonzero(edges)
            total_pixels = edges.shape[0] * edges.shape[1]
            complexity_map[i, j] = edge_pixels / total_pixels if total_pixels > 0 else 0.0

    # Resize heatmap to match original image dimensions
    heatmap_resized = cv2.resize(complexity_map, (w, h), interpolation=cv2.INTER_LINEAR)

    # Normalize to 0-255 and apply colormap
    heatmap_normalized = (heatmap_resized * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_HOT)

    # Blend with original image (50% opacity overlay)
    blended = cv2.addWeighted(img_bgr, 0.5, heatmap_colored, 0.5, 0)

    ok, buf = cv2.imencode(".png", blended)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode complexity heatmap as PNG.",
        )

    data = buf.tobytes()
    try:
        cache_path.write_bytes(data)
    except Exception:
        pass
    return data


def _compute_segmentation_overlay_bytes(
    storage_path: str,
    alpha: float = 0.5,
    conf: float = 0.25,
    show_labels: bool = True,
    overlay_type: str = "panoptic",  # "semantic" | "panoptic"
) -> bytes:
    """Compute a segmentation overlay PNG using OneFormerVisualizer.

    Runs the full OneFormer semantic+panoptic merge pipeline and returns either:
      - "panoptic"  : labeled instance overlay (class + confidence per segment)
      - "semantic"  : flat class-coloured mask overlay (no labels)

    Supports both local file paths and remote URLs.

    Args:
        storage_path : Path to image (local) or URL
        alpha        : Mask transparency (0.0–1.0)
        conf         : Minimum detection confidence for display (0.0–1.0)
        show_labels  : Whether to draw class labels (panoptic mode only)
        overlay_type : "panoptic" (default) or "semantic"

    Returns:
        PNG image bytes with segmentation overlay
    """
    if cv2 is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="cv2 (OpenCV) is not available; cannot compute segmentation.",
        )

    cache_root      = os.getenv("IMAGE_SEGMENTATION_CACHE_ROOT") or os.path.join(
        "backend", "data", "debug_segmentation"
    )
    cache_root_path = Path(cache_root)
    cache_root_path.mkdir(parents=True, exist_ok=True)

    cache_key  = _get_cache_key(storage_path)
    cache_name = (
        f"{cache_key}_seg_{overlay_type}_{int(alpha * 100)}_"
        f"{int(conf * 100)}_{1 if show_labels else 0}.png"
    )
    cache_path = cache_root_path / cache_name

    if cache_path.is_file():
        try:
            return cache_path.read_bytes()
        except Exception:
            pass

    img_bgr = _load_image_from_url_or_path(storage_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    frame = AnalysisFrame(image_id=-1, original_image=img_rgb)

    try:
        SegmentationAnalyzer.analyze(frame, use_semantic=True, use_panoptic=True)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Segmentation failed: {str(e)}. Ensure torch/transformers is installed.",
        )

    # Use OneFormerVisualizer overlay methods via SegmentationAnalyzer
    overlay_rgb = SegmentationAnalyzer.get_segmentation_overlay(
        frame, alpha=alpha, overlay_type=overlay_type
    )

    if overlay_rgb is None:
        # No detections — annotate original
        overlay_rgb = img_rgb.copy()
        cv2.putText(
            overlay_rgb, "No objects detected",
            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2,
        )

    # Add summary footer (objects + coverage)
    total_objects  = frame.attributes.get("segmentation.total_objects", 0)
    scene_coverage = frame.attributes.get("segmentation.scene_coverage", 0.0)
    summary        = f"OneFormer {overlay_type} | Objects: {total_objects} | Coverage: {scene_coverage:.1%}"
    (sum_w, sum_h), sum_bl = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(
        overlay_rgb,
        (5, overlay_rgb.shape[0] - sum_h - sum_bl - 14),
        (sum_w + 14, overlay_rgb.shape[0] - 4),
        (0, 0, 0), -1,
    )
    cv2.putText(
        overlay_rgb, summary,
        (10, overlay_rgb.shape[0] - 8),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2,
    )

    # Back to BGR for imencode
    overlay_bgr = cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", overlay_bgr)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode segmentation overlay as PNG.",
        )

    data = buf.tobytes()
    try:
        cache_path.write_bytes(data)
    except Exception:
        pass
    return data


def _compute_panoptic_overlay_bytes(storage_path: str, alpha: float = 0.4) -> bytes:
    """Convenience wrapper: panoptic (labeled) overlay via OneFormerVisualizer."""
    return _compute_segmentation_overlay_bytes(
        storage_path, alpha=alpha, conf=0.0, show_labels=True, overlay_type="panoptic"
    )


def _compute_depth_map_bytes(path: Path) -> bytes:
    """Compute a depth-map PNG for the given image path.

    This uses the DepthAnalyzer's monocular depth model if it is configured
    (via DEPTH_ANYTHING_ONNX_PATH and onnxruntime). If depth inference is
    not available, we surface a 503 so that the frontend can show a clear
    maintenance overlay rather than silently failing.

    The returned PNG is a single-channel grayscale image where lighter
    pixels correspond to *farther* regions and darker pixels to nearer
    regions, after a simple per-image normalisation.
    """
    if cv2 is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="cv2 (OpenCV) is not available; cannot compute depth maps.",
        )

    # Compute cache path
    cache_root = os.getenv("IMAGE_DEPTH_DEBUG_CACHE_ROOT") or os.path.join("backend", "data", "debug_depth")
    cache_root_path = Path(cache_root)
    cache_root_path.mkdir(parents=True, exist_ok=True)

    cache_name = f"{path.stem}_depth.png"
    cache_path = cache_root_path / cache_name

    if cache_path.is_file():
        try:
            return cache_path.read_bytes()
        except Exception:
            # Fall through to recomputation on any read error
            pass

    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not read image from storage: {path}",
        )

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Minimal AnalysisFrame: we only need original_image and a dummy id.
    frame = AnalysisFrame(image_id=-1, original_image=img_rgb)

    depth = DepthAnalyzer._compute_depth_map(frame)
    if depth is None:
        # Surface as a 503 so clients know depth debug is temporarily
        # unavailable (e.g. missing model weights or onnxruntime).
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Depth debug is not configured. Ensure DEPTH_ANYTHING_ONNX_PATH "
                "is set and onnxruntime is installed."
            ),
        )

    import numpy as _np  # local alias to keep debug module dependency-light

    arr = _np.asarray(depth, dtype="float32")
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.ndim != 2 or arr.size == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Depth model returned an invalid depth map.",
        )

    # Normalise depth to [0, 1] per-image to maximise visual contrast.
    d_min = float(_np.nanmin(arr))
    d_max = float(_np.nanmax(arr))
    if not _np.isfinite(d_min) or not _np.isfinite(d_max):
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Depth map contained only NaN / Inf values.",
        )

    if d_max > d_min:
        norm = (arr - d_min) / (d_max - d_min)
    else:
        norm = _np.zeros_like(arr, dtype="float32")

    norm = _np.clip(norm, 0.0, 1.0)
    depth_uint8 = (norm * 255.0).astype("uint8")

    ok, buf = cv2.imencode(".png", depth_uint8)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode depth map as PNG.",
        )

    data = buf.tobytes()
    try:
        cache_path.write_bytes(data)
    except Exception:
        # Cache write failure should not break the endpoint
        pass
    return data


@router.get("/images/{image_id}/edges", summary="Return edge-map debug view for an image")
def get_image_edge_map(
    image_id: int,
    t1: int = 50,
    t2: int = 150,
    l2: bool = True,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_tagger),
) -> Response:
    """Serve a PNG edge map for the requested image.

    This endpoint is intended purely for *debug / teaching* purposes. It
    allows Explorer (and other tools) to show "what the algorithm sees"
    when computing complexity and related metrics.
    
    Supports both local file paths and remote URLs.
    """
    image: Optional[Image] = db.query(Image).filter(Image.id == image_id).first()
    if image is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    storage_path = getattr(image, "storage_path", None)
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image has no storage_path configured",
        )

    data = _compute_edge_map_bytes(storage_path, t1=t1, t2=t2, l2=l2)
    return Response(content=data, media_type="image/png")


@router.get("/images/{image_id}/depth", summary="Return depth-map debug view for an image")
def get_image_depth_map(
    image_id: int,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_tagger),
) -> Response:
    """Serve a PNG depth map for the requested image.

    This endpoint is intended purely for *debug / teaching* purposes. It
    exposes the monocular depth prediction used by the spatial metrics so
    that students can see "what the model thinks is near vs far".

    If the depth model is not configured, the endpoint returns HTTP 503 so
    that the frontend can surface a clear maintenance overlay instead of
    a generic network error.
    """
    from backend.models.assets import Image  # local import to avoid circularity

    session: Session = db
    image = session.query(Image).filter(Image.id == image_id).one_or_none()
    if image is None or not image.storage_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image {image_id} not found or has no storage_path.",
        )

    path = _resolve_image_path(image.storage_path)

    data = _compute_depth_map_bytes(path)
    return Response(content=data, media_type="image/png")
@router.get("/images/{image_id}/complexity", summary="Return complexity heatmap debug view for an image")
def get_image_complexity_heatmap(
    image_id: int,
    patch_size: int = 64,
    stride: int = 32,
    t1: int = 50,
    t2: int = 150,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_tagger),
) -> Response:
    """Serve a PNG complexity heatmap for the requested image.

    This endpoint shows regionalized edge density across the image:
    - Each patch is analyzed using Canny edge detection
    - complexity_score = edge_pixels / total_pixels
    - Results are displayed as a heatmap overlay (red=high, dark=low)

    Supports both local file paths and remote URLs.

    Parameters:
    - patch_size: Size of each analysis region (default 64)
    - stride: Step size between regions (default 32)
    - t1, t2: Canny edge detection thresholds
    """
    image: Optional[Image] = db.query(Image).filter(Image.id == image_id).first()
    if image is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    storage_path = getattr(image, "storage_path", None)
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image has no storage_path configured",
        )

    data = _compute_complexity_heatmap_bytes(
        storage_path, 
        patch_size=patch_size, 
        stride=stride,
        canny_low=t1,
        canny_high=t2,
    )
    return Response(content=data, media_type="image/png")


@router.get("/images/{image_id}/segmentation", summary="Return semantic segmentation overlay (OneFormerVisualizer)")
def get_image_segmentation(
    image_id: int,
    alpha: float = 0.5,
    conf: float = 0.25,
    labels: bool = True,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_tagger),
) -> Response:
    """Serve a PNG semantic mask overlay produced by OneFormerVisualizer.

    Runs the full semantic + panoptic merge pipeline and returns flat
    class-coloured masks (no per-instance labels).  Use the /segmentation/panoptic
    endpoint for labelled instance overlays.

    Parameters:
    - alpha  : Mask transparency (0.0-1.0, default 0.5)
    - conf   : Minimum detection confidence (0.0-1.0, default 0.25)
    - labels : Ignored for semantic mode; kept for API compatibility
    """
    image: Optional[Image] = db.query(Image).filter(Image.id == image_id).first()
    if image is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    storage_path = getattr(image, "storage_path", None)
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image has no storage_path configured",
        )

    data = _compute_segmentation_overlay_bytes(
        storage_path, alpha=alpha, conf=conf, show_labels=False, overlay_type="semantic"
    )
    return Response(content=data, media_type="image/png")


@router.get(
    "/images/{image_id}/segmentation/panoptic",
    summary="Return panoptic (labeled instance) overlay via OneFormerVisualizer",
)
def get_image_segmentation_panoptic(
    image_id: int,
    alpha: float = 0.4,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_tagger),
) -> Response:
    """Serve a PNG panoptic overlay: merged semantic-panoptic instance masks with labels.

    Each segment is annotated with its class name and confidence score.
    This view corresponds directly to Panel 2 ("Labeled Overlay") of the
    OneFormerVisualizer 4-panel figure.

    Parameters:
    - alpha : Mask transparency (0.0-1.0, default 0.4)
    """
    image: Optional[Image] = db.query(Image).filter(Image.id == image_id).first()
    if image is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    storage_path = getattr(image, "storage_path", None)
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image has no storage_path configured",
        )

    data = _compute_panoptic_overlay_bytes(storage_path, alpha=alpha)
    return Response(content=data, media_type="image/png")


def _compute_room_detection_overlay_bytes(
    storage_path: str,
) -> bytes:
    """Compute a room detection overlay PNG for the given image.

    Uses Places365 classifier to identify room type and displays the
    classification results overlaid on the original image.
    
    Args:
        storage_path: Path to image (local or URL)
        
    Returns:
        PNG image bytes with room detection overlay
    """
    if cv2 is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="cv2 (OpenCV) is not available; cannot compute room detection.",
        )

    # Load image from URL or local path (BGR format)
    img_bgr = _load_image_from_url_or_path(storage_path)
    
    # Convert to RGB for analysis
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Create analysis frame and run room detection
    frame = AnalysisFrame(image_id=-1, original_image=img_rgb)
    
    try:
        result = RoomDetectionAnalyzer.analyze(frame, top_k=5, apply_object_consistency=False)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Room detection failed: {str(e)}. Ensure torch/torchvision is installed.",
        )
    
    # Create overlay
    overlay = img_bgr.copy()
    h, w = overlay.shape[:2]
    
    # Semi-transparent background for text
    panel_height = 180
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)  # Dark gray
    
    # Blend panel at bottom
    overlay[-panel_height:] = cv2.addWeighted(
        overlay[-panel_height:], 0.3, panel, 0.7, 0
    )
    
    # Draw coarse prediction (main result)
    top_coarse = result.get("top_coarse", {})
    coarse_label = top_coarse.get("label", "unknown")
    coarse_prob = top_coarse.get("probability", 0.0)
    
    text = f"Room Type: {coarse_label.upper()}"
    cv2.putText(overlay, text, (15, h - panel_height + 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    
    conf_text = f"Confidence: {coarse_prob:.1%}"
    cv2.putText(overlay, conf_text, (15, h - panel_height + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    
    # Draw fine-grained top-5
    cv2.putText(overlay, "Fine-grained predictions:", (15, h - panel_height + 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
    
    fine_preds = result.get("room_type_fine", [])[:5]
    y_offset = h - panel_height + 110
    for i, (label, prob) in enumerate(fine_preds):
        text = f"{i+1}. {label}: {prob:.1%}"
        cv2.putText(overlay, text, (25, y_offset + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    # Draw coarse distribution bar chart on the right
    coarse_probs = result.get("room_type_coarse", {})
    sorted_coarse = sorted(coarse_probs.items(), key=lambda x: x[1], reverse=True)[:6]
    
    bar_x = w - 250
    bar_y = h - panel_height + 25
    bar_width = 200
    bar_height = 15
    
    cv2.putText(overlay, "Coarse distribution:", (bar_x, bar_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)
    
    for i, (cat, prob) in enumerate(sorted_coarse):
        y = bar_y + 10 + i * 22
        # Draw bar background
        cv2.rectangle(overlay, (bar_x, y), (bar_x + bar_width, y + bar_height),
                      (60, 60, 60), -1)
        # Draw bar fill
        fill_width = int(bar_width * prob)
        color = (100, 200, 100) if i == 0 else (100, 150, 200)
        cv2.rectangle(overlay, (bar_x, y), (bar_x + fill_width, y + bar_height),
                      color, -1)
        # Draw label
        cv2.putText(overlay, f"{cat[:10]}: {prob:.0%}", (bar_x + 5, y + 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)

    ok, buf = cv2.imencode(".png", overlay)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode room detection overlay as PNG.",
        )

    return buf.tobytes()


@router.get("/images/{image_id}/room", summary="Return room type detection overlay for an image")
def get_image_room_detection(
    image_id: int,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_tagger),
) -> Response:
    """Serve a PNG room type detection overlay for the requested image.

    This endpoint uses Places365 classifier to identify the room type,
    then displays the results overlaid on the original image.

    The overlay includes:
    - Primary coarse room type (bathroom, bedroom, kitchen, etc.)
    - Confidence score
    - Top-5 fine-grained Places365 predictions
    - Coarse category distribution bar chart

    Supports both local file paths and remote URLs.
    """
    image: Optional[Image] = db.query(Image).filter(Image.id == image_id).first()
    if image is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    storage_path = getattr(image, "storage_path", None)
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image has no storage_path configured",
        )

    data = _compute_room_detection_overlay_bytes(storage_path)
    return Response(content=data, media_type="image/png")


def _compute_material_detection_overlay_bytes(
    storage_path: str,
) -> bytes:
    """Compute a material detection overlay PNG for the given image.

    Uses Gemini Flash VLM to identify materials, finishes, and textures,
    then displays the results overlaid on the original image.

    Args:
        storage_path: Path to image (local or URL)

    Returns:
        PNG image bytes with material detection overlay
    """
    if cv2 is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="cv2 (OpenCV) is not available; cannot compute material detection.",
        )

    # Check cache first
    cache_root = os.getenv("IMAGE_MATERIALS_CACHE_ROOT") or os.path.join("backend", "data", "debug_materials")
    cache_root_path = Path(cache_root)
    cache_root_path.mkdir(parents=True, exist_ok=True)

    cache_key = _get_cache_key(storage_path)
    cache_name = f"{cache_key}_materials.png"
    cache_path = cache_root_path / cache_name

    if cache_path.is_file():
        try:
            return cache_path.read_bytes()
        except Exception:
            pass

    # Load image from URL or local path (BGR format)
    img_bgr = _load_image_from_url_or_path(storage_path)

    # Convert to RGB for analysis
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Create analysis frame and run material detection
    frame = AnalysisFrame(image_id=-1, original_image=img_rgb)

    try:
        result = GeminiMaterialAnalyzer.analyze(frame)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Material detection failed: {str(e)}. Ensure GEMINI_API_KEY is set.",
        )

    # Create overlay
    overlay = img_bgr.copy()
    h, w = overlay.shape[:2]

    is_stub = result.get("stub", False)
    error_msg = result.get("error", "")

    if is_stub:
        # VLM not configured or API error - show a helpful message
        panel_height = 100
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 60)
        overlay[-panel_height:] = cv2.addWeighted(
            overlay[-panel_height:], 0.3, panel, 0.7, 0
        )

        if "RESOURCE_EXHAUSTED" in str(error_msg) or "429" in str(error_msg):
            cv2.putText(overlay, "Material Detection: API Quota Exhausted",
                         (15, h - panel_height + 28),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 150, 255), 2)
            cv2.putText(overlay, "Gemini API free-tier rate limit reached. Wait or upgrade billing.",
                         (15, h - panel_height + 55),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 200), 1)
            cv2.putText(overlay, "See: https://ai.google.dev/gemini-api/docs/rate-limits",
                         (15, h - panel_height + 78),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 140, 170), 1)
        elif error_msg and result.get("engine") != "StubEngine":
            cv2.putText(overlay, "Material Detection: API Error",
                         (15, h - panel_height + 28),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 150, 255), 2)
            # Truncate error for display
            short_err = str(error_msg)[:90]
            cv2.putText(overlay, short_err,
                         (15, h - panel_height + 55),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 200), 1)
        else:
            cv2.putText(overlay, "Material Detection: No VLM Configured",
                         (15, h - panel_height + 28),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 150, 255), 2)
            cv2.putText(overlay, "Set GEMINI_API_KEY to enable Gemini Flash material analysis",
                         (15, h - panel_height + 55),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 200), 1)

        # Do NOT cache error/stub results so they refresh on retry
        ok, buf = cv2.imencode(".png", overlay)
        if not ok:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to encode material detection overlay as PNG.",
            )
        return buf.tobytes()
    else:
        # Full material detection result
        materials = result.get("materials", [])
        dominant = result.get("dominant_material", "unknown")
        palette = result.get("material_palette", [])
        style_note = result.get("style_note", "")

        # Calculate panel height based on content
        num_materials = min(len(materials), 8)
        panel_height = max(180, 70 + num_materials * 24 + 40)
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (35, 35, 45)

        overlay[-panel_height:] = cv2.addWeighted(
            overlay[-panel_height:], 0.25, panel, 0.75, 0
        )

        # Title
        cv2.putText(overlay, f"Dominant: {dominant.upper()}",
                     (15, h - panel_height + 28),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        # Palette
        if palette:
            palette_text = "Palette: " + ", ".join(palette[:5])
            cv2.putText(overlay, palette_text,
                         (15, h - panel_height + 52),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 200, 180), 1)

        # Material details
        y_start = h - panel_height + 75
        for i, mat in enumerate(materials[:8]):
            mat_name = mat.get("material", "unknown")
            location = mat.get("location", "")
            coverage = mat.get("coverage", 0.0)
            confidence = mat.get("confidence", 0.0)
            finish = mat.get("finish", "")

            # Coverage bar
            bar_x = 15
            bar_y = y_start + i * 24
            bar_width = 120
            bar_height = 14

            # Bar background
            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + bar_width, bar_y + bar_height),
                          (60, 60, 70), -1)

            # Bar fill based on coverage
            fill_width = int(bar_width * min(coverage, 1.0))
            # Color based on confidence
            if confidence >= 0.8:
                bar_color = (100, 220, 100)  # Green for high confidence
            elif confidence >= 0.5:
                bar_color = (100, 180, 220)  # Blue for medium
            else:
                bar_color = (180, 140, 100)  # Orange for lower

            cv2.rectangle(overlay, (bar_x, bar_y), (bar_x + fill_width, bar_y + bar_height),
                          bar_color, -1)

            # Material label
            conf_pct = int(confidence * 100)
            cov_pct = int(coverage * 100)
            label = f"{mat_name}"
            if finish and finish not in ("", "unknown", "None"):
                label += f" ({finish})"
            label += f" - {location}" if location else ""

            cv2.putText(overlay, label,
                         (bar_x + bar_width + 10, bar_y + 11),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220, 220, 220), 1)

            # Coverage/confidence text on bar
            cv2.putText(overlay, f"{cov_pct}%",
                         (bar_x + 3, bar_y + 11),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255, 255, 255), 1)

        # Style note at the bottom
        if style_note:
            style_y = h - 15
            # Truncate if too long
            if len(style_note) > 80:
                style_note = style_note[:77] + "..."
            cv2.putText(overlay, style_note,
                         (15, style_y),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.38, (160, 160, 180), 1)

        # Engine label in top-right
        engine_name = result.get("engine", "Gemini Flash")
        cv2.putText(overlay, f"VLM: {engine_name}",
                     (w - 200, h - panel_height + 20),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.35, (140, 140, 160), 1)

    ok, buf = cv2.imencode(".png", overlay)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode material detection overlay as PNG.",
        )

    data = buf.tobytes()
    try:
        cache_path.write_bytes(data)
    except Exception:
        pass
    return data


@router.get("/images/{image_id}/materials", summary="Return material detection overlay for an image")
def get_image_materials(
    image_id: int,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_tagger),
) -> Response:
    """Serve a PNG material detection overlay for the requested image.

    This endpoint uses Gemini Flash VLM to identify materials, finishes,
    and textures in the image, then displays the results overlaid on
    the original image.

    The overlay includes:
    - Dominant material identification
    - Material palette summary
    - Individual material coverage bars with confidence
    - Finish and location details
    - Style note describing the overall material palette

    Requires GEMINI_API_KEY environment variable to be set.
    Falls back to a helpful message when no VLM is configured.

    Supports both local file paths and remote URLs.
    """
    image: Optional[Image] = db.query(Image).filter(Image.id == image_id).first()
    if image is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    storage_path = getattr(image, "storage_path", None)
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image has no storage_path configured",
        )

    data = _compute_material_detection_overlay_bytes(storage_path)
    return Response(content=data, media_type="image/png")


@router.get("/images/{image_id}/materials/json", summary="Return material detection JSON for an image")
def get_image_materials_json(
    image_id: int,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_tagger),
) -> dict:
    """Return the raw material detection JSON for the requested image.

    Useful for programmatic access to material detection results without
    the overlay visualization.

    Requires GEMINI_API_KEY environment variable to be set.
    """
    image: Optional[Image] = db.query(Image).filter(Image.id == image_id).first()
    if image is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    storage_path = getattr(image, "storage_path", None)
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image has no storage_path configured",
        )

    # Load image
    img_bgr = _load_image_from_url_or_path(storage_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Run analysis
    frame = AnalysisFrame(image_id=image_id, original_image=img_rgb)

    try:
        result = GeminiMaterialAnalyzer.analyze(frame)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Material detection failed: {str(e)}",
        )

    return result


@router.get("/pipeline_health")
def pipeline_health() -> dict:
    """Return a lightweight view of the science pipeline health.

    This avoids hitting the database and instead instantiates the
    configured analyzers directly, grouping them by tier and
    reporting their requires/provides contracts.
    """
    summary: dict = {
        "import_ok": True,
        "cv2_available": getattr(science_pipeline, "cv2", None) is not None,
        "analyzers_by_tier": {},
        "warnings": [],
    }

    analyzer_class_names = [
        "ColorAnalyzer",
        "ComplexityAnalyzer",
        "TextureAnalyzer",
        "FractalAnalyzer",
        "SymmetryAnalyzer",
        "NaturalnessAnalyzer",
        "DepthAnalyzer",
        "CognitiveStateAnalyzer",
        "SegmentationAnalyzer",
        "RoomDetectionAnalyzer",
        "GeminiMaterialAnalyzer",
    ]

    analyzer_classes = []
    for name in analyzer_class_names:
        cls = getattr(science_pipeline, name, None)
        if cls is None:
            summary["warnings"].append(f"Analyzer class {name} missing from pipeline module.")
            continue
        analyzer_classes.append(cls)

    for cls in analyzer_classes:
        try:
            inst = cls()
        except Exception as exc:  # pragma: no cover - defensive
            summary.setdefault("analyzer_errors", []).append(
                {"analyzer": cls.__name__, "error": repr(exc)}
            )
            continue
        tier = getattr(inst, "tier", "unknown")
        requires = list(getattr(inst, "requires", []))
        provides = list(getattr(inst, "provides", []))
        entry = {
            "name": getattr(inst, "name", cls.__name__),
            "tier": tier,
            "requires": requires,
            "provides": provides,
        }
        summary["analyzers_by_tier"].setdefault(tier, []).append(entry)

    return summary


# =============================================================================
# /materials2  — OneFormer + SigLIP2 (clip_material) pipeline
# =============================================================================

def _compute_materials2_overlay_bytes(storage_path: str) -> bytes:
    """
    Run the MaterialIdentificationPipeline and render a confidence overlay PNG.

    Layout:
      - Coloured instance masks (same palette as segmentation panoptic overlay)
      - Per-instance label: class name / top material / confidence bar
      - Footer: total instances, indeterminate count
    """
    if cv2 is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="cv2 (OpenCV) is not available.",
        )

    cache_root      = os.getenv("IMAGE_MATERIALS2_CACHE_ROOT") or os.path.join(
        "backend", "data", "debug_materials2"
    )
    cache_root_path = Path(cache_root)
    cache_root_path.mkdir(parents=True, exist_ok=True)

    cache_key  = _get_cache_key(storage_path)
    cache_name = f"{cache_key}_materials2.png"
    cache_path = cache_root_path / cache_name

    if cache_path.is_file():
        try:
            return cache_path.read_bytes()
        except Exception:
            pass

    img_bgr = _load_image_from_url_or_path(storage_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        from PIL import Image as PILImage
        pipeline = _get_materials2_pipeline()
        results  = pipeline.run(PILImage.fromarray(img_rgb), show_voting_report=False)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Material identification pipeline failed: {str(e)}",
        )

    overlay = img_rgb.copy()
    h, w    = overlay.shape[:2]

    # Assign a consistent colour per class_id
    np.random.seed(42)
    class_colors: dict = {}

    for r in results:
        cls_id = r["class_id"]
        if cls_id not in class_colors:
            rng = np.random.default_rng(abs(hash(r["class_name"])) % (2 ** 32))
            class_colors[cls_id] = tuple(int(c) for c in rng.integers(120, 240, size=3))
        color    = class_colors[cls_id]
        mask     = r["mask"]
        top_mat  = r["top_material"]
        top_sc   = r["top_score"]
        is_indet = top_mat == "Indeterminate Material"

        # Semi-transparent fill
        colored              = np.zeros_like(overlay)
        colored[mask]        = color
        overlay              = cv2.addWeighted(overlay, 1.0, colored, 0.45, 0)

        # Label block centred on mask
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())

        cls_label = r["class_name"]
        mat_label = top_mat if not is_indet else "Indeterminate"
        conf_pct  = f"{top_sc:.0%}" if not is_indet else "—"
        line1     = f"#{r['instance_idx']} {cls_label}"
        line2     = f"{mat_label} {conf_pct}"

        font      = cv2.FONT_HERSHEY_SIMPLEX
        scale     = 0.42
        thick     = 1
        pad       = 4

        (w1, h1), _ = cv2.getTextSize(line1, font, scale, thick)
        (w2, h2), _ = cv2.getTextSize(line2, font, scale, thick)
        bw = max(w1, w2) + pad * 2
        bh = h1 + h2 + pad * 3

        bx = max(0, cx - bw // 2)
        by = max(0, cy - bh // 2)

        bg_color = (40, 40, 40) if is_indet else tuple(int(c * 0.65) for c in color)
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), bg_color, -1)
        cv2.rectangle(overlay, (bx, by), (bx + bw, by + bh), color, 1)

        cv2.putText(overlay, line1, (bx + pad, by + pad + h1),
                    font, scale, (255, 255, 255), thick)

        # Confidence bar behind line 2
        bar_y = by + pad * 2 + h1
        if not is_indet:
            bar_len = int((bw - pad * 2) * min(top_sc, 1.0))
            cv2.rectangle(overlay, (bx + pad, bar_y),
                          (bx + pad + bar_len, bar_y + h2), color, -1)
        cv2.putText(overlay, line2, (bx + pad, bar_y + h2),
                    font, scale, (255, 255, 255), thick)

    # Footer
    total    = len(results)
    indet    = sum(1 for r in results if r["top_material"] == "Indeterminate Material")
    footer   = f"OneFormer + SigLIP2 | {total} instances | {indet} indeterminate"
    (fw, fh), fbl = cv2.getTextSize(footer, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(overlay, (4, h - fh - fbl - 12), (fw + 14, h - 4), (0, 0, 0), -1)
    cv2.putText(overlay, footer, (9, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    ok, buf = cv2.imencode(".png", overlay_bgr)
    if not ok:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to encode materials2 overlay as PNG.",
        )

    data = buf.tobytes()
    try:
        cache_path.write_bytes(data)
    except Exception:
        pass
    return data


@router.get(
    "/images/{image_id}/materials2",
    summary="Return OneFormer + SigLIP2 material confidence overlay",
)
def get_image_materials2(
    image_id: int,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_tagger),
) -> Response:
    """Serve a PNG overlay showing per-instance material predictions from the
    OneFormer + SigLIP2 pipeline.

    Each detected segment is annotated with:
      - Instance index and semantic class name
      - Top predicted material and SigLIP2 confidence score
      - A confidence bar proportional to the sigmoid score
      - Indeterminate flag when the top-2 margin is below threshold

    The footer shows the pipeline name and a summary of indeterminate instances.

    Requires torch + transformers (OneFormer) and a SigLIP2-compatible
    transformers build (google/siglip2-so400m-patch16-naflex).
    """
    image: Optional[Image] = db.query(Image).filter(Image.id == image_id).first()
    if image is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    storage_path = getattr(image, "storage_path", None)
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image has no storage_path configured",
        )

    data = _compute_materials2_overlay_bytes(storage_path)
    return Response(content=data, media_type="image/png")


@router.get(
    "/images/{image_id}/materials2/json",
    summary="Return raw JSON from OneFormer + SigLIP2 material identification",
)
def get_image_materials2_json(
    image_id: int,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_tagger),
) -> list:
    """Return the full per-instance material identification results as JSON.

    Each item in the returned list contains:
      instance_idx, class_id, class_name, seg_confidence,
      material_group, top_material, top_score, margin,
      material_scores (top-10 candidates with scores),
      vote_source (set if spatial voting overrode an Indeterminate result)

    Masks and PIL crops are stripped for JSON serialisability.
    """
    image: Optional[Image] = db.query(Image).filter(Image.id == image_id).first()
    if image is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    storage_path = getattr(image, "storage_path", None)
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image has no storage_path configured",
        )

    img_bgr = _load_image_from_url_or_path(storage_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    try:
        from PIL import Image as PILImage
        pipeline = _get_materials2_pipeline()
        results  = pipeline.run(PILImage.fromarray(img_rgb), show_voting_report=False)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Material identification pipeline failed: {str(e)}",
        )

    return MaterialIdentificationPipeline.to_json_safe(results)


# ── Affordance Prediction ────────────────────────────────────────────────────

_AFFORDANCE_ANALYZER: Optional[AffordanceAnalyzer] = None


@router.get(
    "/images/{image_id}/affordance",
    summary="Predict environmental affordance scores for an image",
)
def get_image_affordance_scores(
    image_id: int,
    db: Session = Depends(get_db),
    user: CurrentUser = Depends(require_tagger),
):
    """Run the affordance prediction pipeline on the requested image.

    Returns a JSON object with suitability scores (1–7 Likert scale) for
    5 environmental activities: Sleep, Cook, Computer Work, Conversation,
    and Yoga/Stretching.

    Requires trained LightGBM models in backend/science/data/affordance_models/.
    Run ``python -m backend.scripts.train_affordance_models`` to train them.
    """
    image: Optional[Image] = db.query(Image).filter(Image.id == image_id).first()
    if image is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Image not found")

    storage_path = getattr(image, "storage_path", None)
    if not storage_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Image has no storage_path configured",
        )

    img_bgr = _load_image_from_url_or_path(storage_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    frame = AnalysisFrame(image_id=image_id, original_image=img_rgb)

    # Run segmentation (required for affordance features)
    SegmentationAnalyzer.analyze(frame, use_semantic=True, use_panoptic=True)

    # Run affordance prediction
    global _AFFORDANCE_ANALYZER
    if _AFFORDANCE_ANALYZER is None:
        _AFFORDANCE_ANALYZER = AffordanceAnalyzer()
    _AFFORDANCE_ANALYZER.analyze(frame)

    # Build response
    scores = {}
    for aff_id in AFFORDANCE_IDS:
        score = frame.attributes.get(f"affordance.{aff_id}")
        norm = frame.attributes.get(f"affordance.{aff_id}_norm")
        if score is not None:
            scores[aff_id] = {
                "name": AFFORDANCE_NAMES.get(aff_id, aff_id),
                "score": round(score, 2),
                "normalized": round(norm, 3) if norm is not None else None,
            }

    return {
        "image_id": image_id,
        "affordances": scores,
        "n_segments": frame.metadata.get("affordance.n_segments", 0),
        "segment_classes": frame.metadata.get("affordance.segment_classes", []),
    }