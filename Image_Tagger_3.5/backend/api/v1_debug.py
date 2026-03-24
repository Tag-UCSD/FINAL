"""
Debug / visualization endpoints for the Explorer UI.

Provides on-the-fly image processing overlays (edges, depth, complexity,
segmentation, room detection, materials) used by the Explorer debug modes.
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional
import hashlib

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    import requests
except Exception:
    requests = None

from fastapi import APIRouter, Depends, HTTPException, Response, status
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
from backend.models.assets import Image

router = APIRouter(prefix="/v1/debug", tags=["Debug / Visualization"])

_MATERIALS2_PIPELINE: Optional[MaterialIdentificationPipeline] = None


def _get_materials2_pipeline() -> MaterialIdentificationPipeline:
    global _MATERIALS2_PIPELINE
    if _MATERIALS2_PIPELINE is None:
        _MATERIALS2_PIPELINE = MaterialIdentificationPipeline.from_pretrained()
    return _MATERIALS2_PIPELINE


def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def _load_image(storage_path: str) -> np.ndarray:
    """Load an image from URL or local path. Returns BGR numpy array."""
    if cv2 is None:
        raise HTTPException(status_code=500, detail="cv2 (OpenCV) is not available.")

    if _is_url(storage_path):
        if requests is None:
            raise HTTPException(status_code=500, detail="requests library not available.")
        try:
            resp = requests.get(storage_path, timeout=10)
            resp.raise_for_status()
            arr = np.frombuffer(resp.content, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise HTTPException(status_code=500, detail=f"Could not decode image from URL: {storage_path}")
            return img
        except requests.RequestException as e:
            raise HTTPException(status_code=500, detail=f"Failed to download image: {e}")
    else:
        path = _resolve_path(storage_path)
        if not path.is_file():
            raise HTTPException(status_code=404, detail=f"Image file not found: {path}")
        img = cv2.imread(str(path))
        if img is None:
            raise HTTPException(status_code=500, detail=f"Could not read image: {path}")
        return img


def _cache_key(storage_path: str) -> str:
    if _is_url(storage_path):
        return hashlib.md5(storage_path.encode()).hexdigest()
    return Path(storage_path).stem


def _resolve_path(storage_path: str) -> Path:
    raw = Path(storage_path)
    if raw.is_file():
        return raw
    root = os.getenv("IMAGE_STORAGE_ROOT")
    if root:
        candidate = Path(root) / storage_path
        if candidate.is_file():
            return candidate
    return raw


def _get_image_or_404(db: Session, image_id: int) -> Image:
    image = db.query(Image).filter(Image.id == image_id).first()
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    if not getattr(image, "storage_path", None):
        raise HTTPException(status_code=404, detail="Image has no storage_path")
    return image


def _cached_png(cache_dir: str, name: str, compute_fn) -> bytes:
    """Check cache, compute if missing, return PNG bytes."""
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    fp = cache_path / name
    if fp.is_file():
        try:
            return fp.read_bytes()
        except Exception:
            pass
    data = compute_fn()
    try:
        fp.write_bytes(data)
    except Exception:
        pass
    return data


def _encode_png(img) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise HTTPException(status_code=500, detail="Failed to encode PNG.")
    return buf.tobytes()


# ── Edge Map ─────────────────────────────────────────────────────────────────

def _compute_edges(storage_path: str, t1: int, t2: int, l2: bool) -> bytes:
    img = _load_image(storage_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, t1, t2, L2gradient=l2)
    return _encode_png(edges)


@router.get("/images/{image_id}/edges")
def get_edge_map(image_id: int, t1: int = 50, t2: int = 150, l2: bool = True,
                 db: Session = Depends(get_db)) -> Response:
    image = _get_image_or_404(db, image_id)
    key = _cache_key(image.storage_path)
    cache_dir = os.getenv("IMAGE_DEBUG_CACHE_ROOT", "backend/data/debug_edges")
    data = _cached_png(cache_dir, f"{key}_edges_{t1}_{t2}_{1 if l2 else 0}.png",
                       lambda: _compute_edges(image.storage_path, t1, t2, l2))
    return Response(content=data, media_type="image/png")


# ── Depth Map ────────────────────────────────────────────────────────────────

def _compute_depth(storage_path: str) -> bytes:
    img_bgr = _load_image(storage_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    frame = AnalysisFrame(image_id=-1, original_image=img_rgb)
    depth = DepthAnalyzer._compute_depth_map(frame)
    if depth is None:
        raise HTTPException(status_code=503, detail="Depth model not configured.")
    arr = np.asarray(depth, dtype="float32")
    if arr.ndim == 3:
        arr = arr[..., 0]
    d_min, d_max = float(np.nanmin(arr)), float(np.nanmax(arr))
    if not np.isfinite(d_min) or not np.isfinite(d_max):
        raise HTTPException(status_code=500, detail="Depth map contained NaN/Inf.")
    norm = (arr - d_min) / (d_max - d_min) if d_max > d_min else np.zeros_like(arr)
    return _encode_png((np.clip(norm, 0, 1) * 255).astype("uint8"))


@router.get("/images/{image_id}/depth")
def get_depth_map(image_id: int, db: Session = Depends(get_db)) -> Response:
    image = _get_image_or_404(db, image_id)
    key = _cache_key(image.storage_path)
    cache_dir = os.getenv("IMAGE_DEPTH_DEBUG_CACHE_ROOT", "backend/data/debug_depth")
    data = _cached_png(cache_dir, f"{key}_depth.png",
                       lambda: _compute_depth(image.storage_path))
    return Response(content=data, media_type="image/png")


# ── Complexity Heatmap ───────────────────────────────────────────────────────

def _compute_complexity(storage_path: str, patch: int, stride: int, t1: int, t2: int) -> bytes:
    img = _load_image(storage_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    out_h = max(1, (h - patch) // stride + 1)
    out_w = max(1, (w - patch) // stride + 1)
    cmap = np.zeros((out_h, out_w), dtype=np.float32)
    for i in range(out_h):
        for j in range(out_w):
            p = gray[i*stride:i*stride+patch, j*stride:j*stride+patch]
            edges = cv2.Canny(p, t1, t2)
            cmap[i, j] = np.count_nonzero(edges) / max(edges.size, 1)
    resized = cv2.resize(cmap, (w, h), interpolation=cv2.INTER_LINEAR)
    colored = cv2.applyColorMap((resized * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    blended = cv2.addWeighted(img, 0.5, colored, 0.5, 0)
    return _encode_png(blended)


@router.get("/images/{image_id}/complexity")
def get_complexity(image_id: int, patch_size: int = 64, stride: int = 32,
                   t1: int = 50, t2: int = 150, db: Session = Depends(get_db)) -> Response:
    image = _get_image_or_404(db, image_id)
    key = _cache_key(image.storage_path)
    cache_dir = os.getenv("IMAGE_COMPLEXITY_CACHE_ROOT", "backend/data/debug_complexity")
    data = _cached_png(cache_dir, f"{key}_complexity_{patch_size}_{stride}_{t1}_{t2}.png",
                       lambda: _compute_complexity(image.storage_path, patch_size, stride, t1, t2))
    return Response(content=data, media_type="image/png")


# ── Segmentation ─────────────────────────────────────────────────────────────

def _compute_segmentation(storage_path: str, alpha: float, conf: float, overlay_type: str) -> bytes:
    img_bgr = _load_image(storage_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    frame = AnalysisFrame(image_id=-1, original_image=img_rgb)
    try:
        SegmentationAnalyzer.analyze(frame, use_semantic=True, use_panoptic=True)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Segmentation failed: {e}")
    overlay_rgb = SegmentationAnalyzer.get_segmentation_overlay(frame, alpha=alpha, overlay_type=overlay_type)
    if overlay_rgb is None:
        overlay_rgb = img_rgb.copy()
        cv2.putText(overlay_rgb, "No objects detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    total = frame.attributes.get("segmentation.total_objects", 0)
    coverage = frame.attributes.get("segmentation.scene_coverage", 0.0)
    summary = f"OneFormer {overlay_type} | Objects: {total} | Coverage: {coverage:.1%}"
    (sw, sh), _ = cv2.getTextSize(summary, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(overlay_rgb, (5, overlay_rgb.shape[0]-sh-18), (sw+14, overlay_rgb.shape[0]-4), (0,0,0), -1)
    cv2.putText(overlay_rgb, summary, (10, overlay_rgb.shape[0]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
    return _encode_png(cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR))


@router.get("/images/{image_id}/segmentation")
def get_segmentation(image_id: int, alpha: float = 0.5, conf: float = 0.25,
                     db: Session = Depends(get_db)) -> Response:
    image = _get_image_or_404(db, image_id)
    key = _cache_key(image.storage_path)
    cache_dir = os.getenv("IMAGE_SEGMENTATION_CACHE_ROOT", "backend/data/debug_segmentation")
    data = _cached_png(cache_dir, f"{key}_seg_semantic_{int(alpha*100)}_{int(conf*100)}.png",
                       lambda: _compute_segmentation(image.storage_path, alpha, conf, "semantic"))
    return Response(content=data, media_type="image/png")


# ── Room Detection ───────────────────────────────────────────────────────────

def _compute_room(storage_path: str) -> bytes:
    img_bgr = _load_image(storage_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    frame = AnalysisFrame(image_id=-1, original_image=img_rgb)
    try:
        result = RoomDetectionAnalyzer.analyze(frame, top_k=5, apply_object_consistency=False)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Room detection failed: {e}")
    overlay = img_bgr.copy()
    h, w = overlay.shape[:2]
    panel_height = 180
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    panel[:] = (40, 40, 40)
    overlay[-panel_height:] = cv2.addWeighted(overlay[-panel_height:], 0.3, panel, 0.7, 0)
    top_coarse = result.get("top_coarse", {})
    cv2.putText(overlay, f"Room Type: {top_coarse.get('label','unknown').upper()}",
                (15, h-panel_height+30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,255,255), 2)
    cv2.putText(overlay, f"Confidence: {top_coarse.get('probability',0):.1%}",
                (15, h-panel_height+60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180,180,180), 1)
    cv2.putText(overlay, "Fine-grained predictions:", (15, h-panel_height+90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150,150,150), 1)
    for i, (label, prob) in enumerate(result.get("room_type_fine", [])[:5]):
        cv2.putText(overlay, f"{i+1}. {label}: {prob:.1%}", (25, h-panel_height+110+i*18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200,200,200), 1)
    coarse_probs = result.get("room_type_coarse", {})
    sorted_coarse = sorted(coarse_probs.items(), key=lambda x: x[1], reverse=True)[:6]
    bar_x = w - 250
    cv2.putText(overlay, "Coarse distribution:", (bar_x, h-panel_height+20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150,150,150), 1)
    for i, (cat, prob) in enumerate(sorted_coarse):
        y = h - panel_height + 35 + i * 22
        cv2.rectangle(overlay, (bar_x, y), (bar_x+200, y+15), (60,60,60), -1)
        cv2.rectangle(overlay, (bar_x, y), (bar_x+int(200*prob), y+15),
                      (100,200,100) if i==0 else (100,150,200), -1)
        cv2.putText(overlay, f"{cat[:10]}: {prob:.0%}", (bar_x+5, y+12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,255), 1)
    return _encode_png(overlay)


@router.get("/images/{image_id}/room")
def get_room(image_id: int, db: Session = Depends(get_db)) -> Response:
    image = _get_image_or_404(db, image_id)
    data = _compute_room(image.storage_path)
    return Response(content=data, media_type="image/png")


# ── Materials (Gemini VLM) ───────────────────────────────────────────────────

def _compute_materials(storage_path: str) -> bytes:
    img_bgr = _load_image(storage_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    frame = AnalysisFrame(image_id=-1, original_image=img_rgb)
    try:
        result = GeminiMaterialAnalyzer.analyze(frame)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Material detection failed: {e}")
    overlay = img_bgr.copy()
    h, w = overlay.shape[:2]
    is_stub = result.get("stub", False)
    if is_stub:
        panel_height = 100
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        panel[:] = (40, 40, 60)
        overlay[-panel_height:] = cv2.addWeighted(overlay[-panel_height:], 0.3, panel, 0.7, 0)
        error_msg = result.get("error", "")
        if "RESOURCE_EXHAUSTED" in str(error_msg) or "429" in str(error_msg):
            cv2.putText(overlay, "Material Detection: API Quota Exhausted",
                        (15, h-panel_height+28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100,150,255), 2)
        else:
            cv2.putText(overlay, "Material Detection: No VLM Configured",
                        (15, h-panel_height+28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100,150,255), 2)
            cv2.putText(overlay, "Set GEMINI_API_KEY to enable", (15, h-panel_height+55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180,180,200), 1)
        return _encode_png(overlay)
    materials = result.get("materials", [])
    dominant = result.get("dominant_material", "unknown")
    palette = result.get("material_palette", [])
    num_mats = min(len(materials), 8)
    panel_height = max(180, 70 + num_mats * 24 + 40)
    panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
    panel[:] = (35, 35, 45)
    overlay[-panel_height:] = cv2.addWeighted(overlay[-panel_height:], 0.25, panel, 0.75, 0)
    cv2.putText(overlay, f"Dominant: {dominant.upper()}", (15, h-panel_height+28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    if palette:
        cv2.putText(overlay, "Palette: " + ", ".join(palette[:5]), (15, h-panel_height+52),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180,200,180), 1)
    y_start = h - panel_height + 75
    for i, mat in enumerate(materials[:8]):
        bar_y = y_start + i * 24
        coverage = mat.get("coverage", 0)
        confidence = mat.get("confidence", 0)
        cv2.rectangle(overlay, (15, bar_y), (135, bar_y+14), (60,60,70), -1)
        fill = int(120 * min(coverage, 1.0))
        color = (100,220,100) if confidence >= 0.8 else (100,180,220) if confidence >= 0.5 else (180,140,100)
        cv2.rectangle(overlay, (15, bar_y), (15+fill, bar_y+14), color, -1)
        label = mat.get("material", "unknown")
        finish = mat.get("finish", "")
        if finish and finish not in ("", "unknown", "None"):
            label += f" ({finish})"
        loc = mat.get("location", "")
        if loc:
            label += f" - {loc}"
        cv2.putText(overlay, label, (145, bar_y+11), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220,220,220), 1)
        cv2.putText(overlay, f"{int(coverage*100)}%", (18, bar_y+11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (255,255,255), 1)
    return _encode_png(overlay)


@router.get("/images/{image_id}/materials")
def get_materials(image_id: int, db: Session = Depends(get_db)) -> Response:
    image = _get_image_or_404(db, image_id)
    key = _cache_key(image.storage_path)
    cache_dir = os.getenv("IMAGE_MATERIALS_CACHE_ROOT", "backend/data/debug_materials")
    data = _cached_png(cache_dir, f"{key}_materials.png",
                       lambda: _compute_materials(image.storage_path))
    return Response(content=data, media_type="image/png")


# ── Materials2 (OneFormer + SigLIP2) ─────────────────────────────────────────

def _compute_materials2(storage_path: str) -> bytes:
    img_bgr = _load_image(storage_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    try:
        from PIL import Image as PILImage
        pipeline = _get_materials2_pipeline()
        results = pipeline.run(PILImage.fromarray(img_rgb), show_voting_report=False)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Material identification failed: {e}")
    overlay = img_rgb.copy()
    h, w = overlay.shape[:2]
    np.random.seed(42)
    class_colors = {}
    for r in results:
        cls_id = r["class_id"]
        if cls_id not in class_colors:
            rng = np.random.default_rng(abs(hash(r["class_name"])) % (2**32))
            class_colors[cls_id] = tuple(int(c) for c in rng.integers(120, 240, size=3))
        color = class_colors[cls_id]
        mask = r["mask"]
        colored = np.zeros_like(overlay)
        colored[mask] = color
        overlay = cv2.addWeighted(overlay, 1.0, colored, 0.45, 0)
        ys, xs = np.where(mask)
        if len(xs) == 0:
            continue
        cx, cy = int(xs.mean()), int(ys.mean())
        top_mat = r["top_material"]
        top_sc = r["top_score"]
        is_indet = top_mat == "Indeterminate Material"
        line1 = f"#{r['instance_idx']} {r['class_name']}"
        line2 = f"{top_mat if not is_indet else 'Indeterminate'} {f'{top_sc:.0%}' if not is_indet else '—'}"
        font, scale, thick, pad = cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1, 4
        (w1, h1), _ = cv2.getTextSize(line1, font, scale, thick)
        (w2, h2), _ = cv2.getTextSize(line2, font, scale, thick)
        bw, bh = max(w1, w2) + pad*2, h1 + h2 + pad*3
        bx, by = max(0, cx - bw//2), max(0, cy - bh//2)
        bg = (40,40,40) if is_indet else tuple(int(c*0.65) for c in color)
        cv2.rectangle(overlay, (bx,by), (bx+bw, by+bh), bg, -1)
        cv2.rectangle(overlay, (bx,by), (bx+bw, by+bh), color, 1)
        cv2.putText(overlay, line1, (bx+pad, by+pad+h1), font, scale, (255,255,255), thick)
        bar_y = by + pad*2 + h1
        if not is_indet:
            bar_len = int((bw - pad*2) * min(top_sc, 1.0))
            cv2.rectangle(overlay, (bx+pad, bar_y), (bx+pad+bar_len, bar_y+h2), color, -1)
        cv2.putText(overlay, line2, (bx+pad, bar_y+h2), font, scale, (255,255,255), thick)
    total = len(results)
    indet = sum(1 for r in results if r["top_material"] == "Indeterminate Material")
    footer = f"OneFormer + SigLIP2 | {total} instances | {indet} indeterminate"
    (fw, fh), _ = cv2.getTextSize(footer, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.rectangle(overlay, (4, h-fh-16), (fw+14, h-4), (0,0,0), -1)
    cv2.putText(overlay, footer, (9, h-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
    return _encode_png(cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


@router.get("/images/{image_id}/materials2")
def get_materials2(image_id: int, db: Session = Depends(get_db)) -> Response:
    image = _get_image_or_404(db, image_id)
    key = _cache_key(image.storage_path)
    cache_dir = os.getenv("IMAGE_MATERIALS2_CACHE_ROOT", "backend/data/debug_materials2")
    data = _cached_png(cache_dir, f"{key}_materials2.png",
                       lambda: _compute_materials2(image.storage_path))
    return Response(content=data, media_type="image/png")


# ── Affordance Prediction ────────────────────────────────────────────────────

_AFFORDANCE_ANALYZER: Optional[AffordanceAnalyzer] = None


@router.get("/images/{image_id}/affordance")
def get_affordance(image_id: int, db: Session = Depends(get_db)):
    """Predict environmental affordance scores for an image."""
    image = _get_image_or_404(db, image_id)
    img_bgr = _load_image(image.storage_path)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    frame = AnalysisFrame(image_id=image_id, original_image=img_rgb)
    SegmentationAnalyzer.analyze(frame, use_semantic=True, use_panoptic=True)
    global _AFFORDANCE_ANALYZER
    if _AFFORDANCE_ANALYZER is None:
        _AFFORDANCE_ANALYZER = AffordanceAnalyzer()
    _AFFORDANCE_ANALYZER.analyze(frame)
    scores = {}
    for aff_id in AFFORDANCE_IDS:
        key = f"affordance.{aff_id}"
        val = frame.attributes.get(key)
        if val is not None:
            scores[AFFORDANCE_NAMES.get(aff_id, aff_id)] = round(float(val), 3)
    return {"image_id": image_id, "affordance_scores": scores}
