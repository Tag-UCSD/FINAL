"""
Explorer / Discovery API (v1).

Endpoints for the Explorer UI: search, export, attributes, image detail, seeding.
"""
from __future__ import annotations

from datetime import datetime, timezone
import os
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import requests
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.core import get_db
from backend.schemas.discovery import (
    SearchQuery, ImageSearchResult, ExportRequest, AttributeRead,
    AttributeValue, HumanValidation, ImageDetailResult, TagInfo, AffordanceScore,
)
from backend.models.attribute import Attribute
from backend.models.assets import Image
from backend.science.context.affordance import (
    AFFORDANCE_IDS,
    AFFORDANCE_NAMES,
    predict_affordances_with_metadata_from_image,
)

router = APIRouter(prefix="/v1/explorer", tags=["explorer"])


_AFFORDANCE_CACHE_KEY = "affordance_runtime_v1"


def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def _resolve_path(storage_path: str) -> Path:
    raw = Path(storage_path)
    if raw.is_file():
        return raw
    root = Path("data_store")
    candidate = root / storage_path
    if candidate.is_file():
        return candidate
    env_root_value = os.getenv("IMAGE_STORAGE_ROOT", "")
    if env_root_value:
        env_root = Path(env_root_value)
        env_candidate = env_root / storage_path
        if env_candidate.is_file():
            return env_candidate
    return raw


def _load_image_rgb(storage_path: str) -> np.ndarray:
    if _is_url(storage_path):
        resp = requests.get(storage_path, timeout=15)
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        path = _resolve_path(storage_path)
        if not path.is_file():
            raise FileNotFoundError(f"Image file not found: {path}")
        bgr = cv2.imread(str(path))
    if bgr is None:
        raise ValueError(f"Could not load image: {storage_path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _score_list_from_map(score_map: dict | None) -> list[AffordanceScore]:
    score_map = score_map or {}
    out: list[AffordanceScore] = []
    for aff_id in AFFORDANCE_IDS:
        if aff_id not in score_map:
            continue
        out.append(AffordanceScore(
            id=aff_id,
            label=AFFORDANCE_NAMES.get(aff_id, aff_id),
            score=float(score_map[aff_id]),
        ))
    return out


def _get_cached_affordance_payload(image: Image) -> dict | None:
    meta = getattr(image, "meta_data", {}) or {}
    payload = meta.get(_AFFORDANCE_CACHE_KEY)
    if isinstance(payload, dict) and isinstance(payload.get("scores"), dict):
        scores = payload.get("scores") or {}
        if any((not isinstance(v, (int, float))) or float(v) < 1.0 or float(v) > 7.0 for v in scores.values()):
            return None
        return payload
    return None


def _compute_and_cache_affordance_payload(image: Image, db: Session) -> dict:
    try:
        rgb = _load_image_rgb(image.storage_path)
        predicted = predict_affordances_with_metadata_from_image(rgb)
        payload = {
            "scores": {k: round(float(v), 3) for k, v in predicted.get("scores", {}).items()},
            "method": predicted.get("method"),
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
    except Exception:
        payload = {
            "scores": {},
            "method": "unavailable",
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
    meta = dict(getattr(image, "meta_data", {}) or {})
    meta[_AFFORDANCE_CACHE_KEY] = payload
    image.meta_data = meta
    db.add(image)
    db.commit()
    db.refresh(image)
    return payload


def _get_or_compute_affordance_payload(image: Image, db: Session) -> dict:
    cached = _get_cached_affordance_payload(image)
    if cached is not None:
        return cached
    return _compute_and_cache_affordance_payload(image, db)


@router.post("/search", response_model=List[ImageSearchResult])
def search_images(payload: SearchQuery, db: Session = Depends(get_db)):
    q = db.query(Image)

    if getattr(payload, "text", None):
        text = f"%{payload.text}%"
        name_col = getattr(Image, "name", None)
        desc_col = getattr(Image, "description", None)
        if name_col is not None and desc_col is not None:
            q = q.filter((name_col.ilike(text)) | (desc_col.ilike(text)))
        elif name_col is not None:
            q = q.filter(name_col.ilike(text))

    page = max(1, getattr(payload, "page", 1))
    page_size = max(1, min(getattr(payload, "page_size", 48), 200))
    offset = (page - 1) * page_size
    q = q.order_by(Image.id).offset(offset).limit(page_size)

    results: List[ImageSearchResult] = []
    for img in q.all():
        image_id = getattr(img, "id", None)
        if image_id is None:
            continue
        storage_path = getattr(img, "storage_path", None)
        if storage_path and storage_path.startswith("http"):
            url = storage_path
        else:
            thumb_name = getattr(img, "thumbnail_path", None) or f"image_{image_id}.jpg"
            url = f"/static/thumbnails/{thumb_name}"
        meta = getattr(img, "meta_data", {}) or {}
        tags = meta.get("tags", []) if isinstance(meta, dict) else []
        affordance_payload = _get_cached_affordance_payload(img)
        results.append(ImageSearchResult(
            id=image_id,
            url=url,
            tags=tags,
            meta_data=meta,
            affordance_scores=_score_list_from_map((affordance_payload or {}).get("scores")),
            affordance_method=(affordance_payload or {}).get("method"),
        ))

    return results


@router.post("/export")
def export_training_data(payload: ExportRequest, db: Session = Depends(get_db)):
    """Export validation records for given image IDs."""
    if not payload.image_ids:
        return []
    from backend.models.annotation import Validation
    from sqlalchemy import select
    stmt = (
        select(Validation, Image)
        .join(Image, Image.id == Validation.image_id)
        .where(Validation.image_id.in_(payload.image_ids))
    )
    rows = db.execute(stmt).all()
    return [
        {
            "image_id": v.image_id,
            "image_filename": img.filename,
            "attribute_key": v.attribute_key,
            "value": float(v.value),
            "user_id": v.user_id,
            "region_id": v.region_id,
            "duration_ms": v.duration_ms,
            "created_at": v.created_at,
            "source": v.source,
        }
        for v, img in rows
    ]


@router.get("/attributes", response_model=List[AttributeRead])
def list_attributes(db: Session = Depends(get_db)):
    attrs = db.query(Attribute).order_by(Attribute.key).all()
    return [AttributeRead.model_validate(attr) for attr in attrs]


@router.get("/images/{image_id}/detail", response_model=ImageDetailResult)
def get_image_detail(image_id: int, db: Session = Depends(get_db)):
    from backend.models.annotation import Validation
    from backend.models.users import User

    image = db.query(Image).filter(Image.id == image_id).first()
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    storage_path = getattr(image, "storage_path", None)
    if storage_path and storage_path.startswith("http"):
        url = storage_path
    else:
        thumb_name = getattr(image, "thumbnail_path", None) or f"image_{image_id}.jpg"
        url = f"/static/thumbnails/{thumb_name}"

    validations = (
        db.query(Validation)
        .filter(Validation.image_id == image_id)
        .order_by(Validation.attribute_key)
        .all()
    )

    attr_registry = {a.key: a for a in db.query(Attribute).all()}
    user_ids = {v.user_id for v in validations if v.user_id is not None}
    user_map = {}
    if user_ids:
        users = db.query(User).filter(User.id.in_(user_ids)).all()
        user_map = {u.id: u.username for u in users}

    science_attributes: list[AttributeValue] = []
    human_validations: list[HumanValidation] = []

    for v in validations:
        attr = attr_registry.get(v.attribute_key)
        if v.source and v.source.startswith("science_pipeline"):
            science_attributes.append(AttributeValue(
                key=v.attribute_key,
                name=attr.name if attr else v.attribute_key,
                category=attr.category if attr else None,
                value=float(v.value),
                source=v.source,
            ))
        else:
            human_validations.append(HumanValidation(
                user_id=v.user_id,
                username=user_map.get(v.user_id) if v.user_id else None,
                attribute_key=v.attribute_key,
                value=float(v.value),
                duration_ms=v.duration_ms if v.duration_ms else None,
                created_at=v.created_at,
            ))

    meta = getattr(image, "meta_data", {}) or {}
    affordance_payload = _get_cached_affordance_payload(image) or {}
    raw_tags = meta.get("tags", []) if isinstance(meta, dict) else []
    filename = getattr(image, "filename", None) or meta.get("filename", f"image_{image_id}")

    _NAMESPACE_SOURCE = {
        "style": "Semantic Tagger · VLM", "cognitive": "Cognitive Analyzer · VLM",
        "color": "Color Analyzer · CIELAB", "texture": "Texture Analyzer · GLCM",
        "fractal": "Fractal Dimension Analyzer", "symmetry": "Symmetry Analyzer",
        "naturalness": "Naturalness Analyzer", "fluency": "Fluency Analyzer",
        "spatial": "Depth / Spatial Analyzer", "segmentation": "Segmentation · OneFormer",
        "science": "Complexity Analyzer · Canny",
    }

    def _key_to_label(key: str) -> str:
        last = key.rsplit(".", 1)[-1]
        return " ".join(w.capitalize() for w in last.split("_"))

    def _source_label_for(key: str) -> str:
        if "room_function" in key:
            return "Room Classifier · VLM"
        return _NAMESPACE_SOURCE.get(key.split(".")[0], "Science Pipeline")

    tag_infos: list[TagInfo] = [
        TagInfo(label=t, source="preloaded", source_label="Imported with dataset")
        for t in raw_tags
    ]

    _TAG_THRESHOLD = 0.5
    _TAG_NAMESPACES = ("style.", "cognitive.")
    preloaded_lower = {t.label.lower() for t in tag_infos}

    for sa in science_attributes:
        is_tag = any(sa.key.startswith(ns) for ns in _TAG_NAMESPACES) or "room_function" in sa.key
        if is_tag and sa.value >= _TAG_THRESHOLD:
            label = _key_to_label(sa.key)
            if label.lower() not in preloaded_lower:
                tag_infos.append(TagInfo(
                    label=label, source="science_pipeline",
                    source_label=_source_label_for(sa.key),
                    confidence=sa.value, attribute_key=sa.key,
                ))

    return ImageDetailResult(
        id=image.id, url=url, filename=filename, tags=tag_infos,
        meta_data=meta,
        affordance_scores=_score_list_from_map((affordance_payload or {}).get("scores")),
        affordance_method=(affordance_payload or {}).get("method"),
        science_attributes=science_attributes,
        human_validations=human_validations,
    )


@router.get("/images/{image_id}/affordance")
def get_image_affordance(image_id: int, db: Session = Depends(get_db)):
    image = db.query(Image).filter(Image.id == image_id).first()
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")
    payload = _get_or_compute_affordance_payload(image, db)
    return {
        "image_id": image_id,
        "affordance_scores": _score_list_from_map(payload.get("scores")),
        "affordance_method": payload.get("method"),
    }


@router.post("/seed")
def seed_sample_images(payload: Optional[dict] = None, db: Session = Depends(get_db)):
    """Seed the DB with bundled sample image URLs."""
    payload = payload or {}
    force = bool(payload.get("force"))
    existing = db.query(Image).count()
    if existing > 0 and not force:
        return {
            "ok": True, "skipped": True,
            "message": f"Database already has {existing} images; pass force=true to override.",
        }
    from backend.scripts.import_images_from_json import import_images
    json_path = Path(__file__).resolve().parents[1] / "database" / "google_images_import.json"
    if not json_path.exists():
        raise HTTPException(status_code=500, detail=f"Seed file not found: {json_path}")
    result = import_images(str(json_path))
    return {"ok": True, "skipped": False, **result}
