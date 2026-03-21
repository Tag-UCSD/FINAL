"""
Explorer / Discovery API (v1).

This module is the canonical adaptor between the Explorer UI and the
database + training export service. It exposes three main endpoints:

- POST /v1/explorer/search
- POST /v1/explorer/export
- GET  /v1/explorer/attributes

All endpoints are RBAC-protected for taggers (and above).
"""

from __future__ import annotations

from typing import List, Optional
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.core import get_db
from backend.services.auth import require_tagger
from backend.services.training_export import TrainingExporter
from backend.schemas.training import TrainingExample
from backend.schemas.discovery import (
    SearchQuery, ImageSearchResult, ExportRequest, AttributeRead,
    AttributeValue, HumanValidation, ImageDetailResult, TagInfo,
)
from backend.models.attribute import Attribute
from backend.models.assets import Image

router = APIRouter(prefix="/v1/explorer", tags=["explorer"])


@router.post("/search", response_model=List[ImageSearchResult])
def search_images(
    payload: SearchQuery,
    db: Session = Depends(get_db),
    user=Depends(require_tagger),
) -> List[ImageSearchResult]:
    """Search for images matching the query.

    This minimal implementation supports:
      - free-text search over image name / description (if available), or
      - returning a simple paginated list when no filter is provided.

    The exact ranking can be improved later; for now we favour determinism
    and well-formed responses over sophistication.
    """
    # Lazy import to avoid circular dependencies
    from backend.models.assets import Image  # type: ignore

    q = db.query(Image)

    if getattr(payload, "text", None):
        text = f"%{payload.text}%"
        # Try to filter by name / description if those fields exist.
        # We guard each attribute access with hasattr to avoid hard failures
        # across slightly different schemas.
        name_col = getattr(Image, "name", None)
        desc_col = getattr(Image, "description", None)
        if name_col is not None and desc_col is not None:
            q = q.filter((name_col.ilike(text)) | (desc_col.ilike(text)))
        elif name_col is not None:
            q = q.filter(name_col.ilike(text))

    # Use page/page_size from schema (frontend sends these)
    page = max(1, getattr(payload, "page", 1))
    page_size = max(1, min(getattr(payload, "page_size", 48), 200))
    offset = (page - 1) * page_size
    q = q.order_by(getattr(Image, "id")).offset(offset).limit(page_size)

    images = q.all()
    results: List[ImageSearchResult] = []

    base_thumb_url = "/static/thumbnails"

    for img in images:
        image_id = getattr(img, "id", None)
        if image_id is None:
            continue
        
        # Get URL from storage_path (external URLs) or build thumbnail path
        storage_path = getattr(img, "storage_path", None)
        if storage_path and storage_path.startswith("http"):
            url = storage_path
        else:
            thumb_name = getattr(img, "thumbnail_path", None) or f"image_{image_id}.jpg"
            url = f"{base_thumb_url}/{thumb_name}"
        
        # Extract tags from meta_data
        meta = getattr(img, "meta_data", {}) or {}
        tags = meta.get("tags", []) if isinstance(meta, dict) else []
        
        res = ImageSearchResult(id=image_id, url=url, tags=tags, meta_data=meta)
        results.append(res)

    return results


@router.post("/export", response_model=List[TrainingExample])
def export_training_data(
    payload: ExportRequest,
    db: Session = Depends(get_db),
    user=Depends(require_tagger),
) -> List[TrainingExample]:
    """Export training examples for the given image IDs.

    This is a thin wrapper around the TrainingExporter service, returning
    a list of TrainingExample objects suitable for downstream model training.
    """
    if not payload.image_ids:
        return []

    exporter = TrainingExporter(db=db)
    try:
        examples = exporter.export_for_images(payload.image_ids)
    except Exception as exc:  # pragma: no cover - defensive logging
        raise HTTPException(status_code=500, detail=f"training export failed: {exc}") from exc

    return examples


@router.get("/attributes", response_model=List[AttributeRead])
def list_attributes(
    db: Session = Depends(get_db),
    user=Depends(require_tagger),
) -> List[AttributeRead]:
    """Return the attribute registry for Explorer filters.

    We expose all Attribute rows, mapping them into AttributeRead records.
    Pydantic's from_attributes=True handles the ORM-to-schema conversion.
    """
    attrs = db.query(Attribute).order_by(Attribute.key).all()
    return [AttributeRead.model_validate(attr) for attr in attrs]


@router.get("/images/{image_id}/detail", response_model=ImageDetailResult)
def get_image_detail(
    image_id: int,
    db: Session = Depends(get_db),
    user=Depends(require_tagger),
) -> ImageDetailResult:
    """Return the full detail payload for the single-image viewer modal.

    Fetches the image record, all Validation rows for that image, and the
    Attribute registry in a single pass. Returns science-pipeline attributes
    and human validations as separate lists, ready for the frontend to render.
    """
    from backend.models.annotation import Validation  # avoid circular import
    from backend.models.users import User

    image = db.query(Image).filter(Image.id == image_id).first()
    if image is None:
        raise HTTPException(status_code=404, detail="Image not found")

    # Build URL (same logic as search_images)
    storage_path = getattr(image, "storage_path", None)
    if storage_path and storage_path.startswith("http"):
        url = storage_path
    else:
        thumb_name = getattr(image, "thumbnail_path", None) or f"image_{image_id}.jpg"
        url = f"/static/thumbnails/{thumb_name}"

    # Fetch all validations for this image
    validations = (
        db.query(Validation)
        .filter(Validation.image_id == image_id)
        .order_by(Validation.attribute_key)
        .all()
    )

    # Load attribute registry for name/category lookup
    attr_registry = {a.key: a for a in db.query(Attribute).all()}

    # Load user map for username lookup (single query for all users needed)
    user_ids = {v.user_id for v in validations if v.user_id is not None}
    if user_ids:
        users = db.query(User).filter(User.id.in_(user_ids)).all()
        user_map = {u.id: u.username for u in users}
    else:
        user_map = {}

    science_attributes: list[AttributeValue] = []
    human_validations: list[HumanValidation] = []

    for v in validations:
        attr = attr_registry.get(v.attribute_key)
        is_science = bool(v.source and v.source.startswith("science_pipeline"))

        if is_science:
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
    raw_tags = meta.get("tags", []) if isinstance(meta, dict) else []
    filename = getattr(image, "filename", None) or meta.get("filename", f"image_{image_id}")

    # ── Build tag provenance ──────────────────────────────────────────────────
    # Map attribute key namespace prefix → human-readable analyzer name.
    _NAMESPACE_SOURCE: dict[str, str] = {
        "style":       "Semantic Tagger · VLM",
        "cognitive":   "Cognitive Analyzer · VLM",
        "color":       "Color Analyzer · CIELAB",
        "texture":     "Texture Analyzer · GLCM",
        "fractal":     "Fractal Dimension Analyzer",
        "symmetry":    "Symmetry Analyzer",
        "naturalness": "Naturalness Analyzer",
        "fluency":     "Fluency Analyzer",
        "spatial":     "Depth / Spatial Analyzer",
        "segmentation":"Segmentation · OneFormer",
        "science":     "Complexity Analyzer · Canny",
    }

    def _key_to_label(key: str) -> str:
        """'style.minimalist' → 'Minimalist',  'spatial.room_function.home_office' → 'Home Office'"""
        last = key.rsplit(".", 1)[-1]
        return " ".join(w.capitalize() for w in last.split("_"))

    def _source_label_for(key: str) -> str:
        if "room_function" in key:
            return "Room Classifier · VLM"
        ns = key.split(".")[0]
        return _NAMESPACE_SOURCE.get(ns, "Science Pipeline")

    # Preloaded tags (from meta_data.tags at import/upload time)
    tag_infos: list[TagInfo] = [
        TagInfo(label=t, source="preloaded", source_label="Imported with dataset")
        for t in raw_tags
    ]

    # Derive additional tags from high-confidence science attributes.
    # We only promote categorical/semantic attributes to tags (style.*, room_function.*, cognitive.*).
    _TAG_THRESHOLD = 0.5
    _TAG_NAMESPACES = ("style.", "cognitive.")
    preloaded_lower = {t.label.lower() for t in tag_infos}

    for sa in science_attributes:
        key = sa.key
        is_tag_attr = (
            any(key.startswith(ns) for ns in _TAG_NAMESPACES)
            or "room_function" in key
        )
        if is_tag_attr and sa.value >= _TAG_THRESHOLD:
            label = _key_to_label(key)
            if label.lower() not in preloaded_lower:
                tag_infos.append(TagInfo(
                    label=label,
                    source="science_pipeline",
                    source_label=_source_label_for(key),
                    confidence=sa.value,
                    attribute_key=key,
                ))

    return ImageDetailResult(
        id=image.id,
        url=url,
        filename=filename,
        tags=tag_infos,
        meta_data=meta,
        science_attributes=science_attributes,
        human_validations=human_validations,
    )


@router.post("/seed")
def seed_sample_images(
    payload: Optional[dict] = None,
    db: Session = Depends(get_db),
    user=Depends(require_tagger),
):
    """Seed the DB with bundled sample image URLs from google_images_import.json.

    This is an explicit, user-initiated action intended for empty or low-data installs.
    """
    payload = payload or {}
    force = bool(payload.get("force"))
    existing = db.query(Image).count()
    if existing > 0 and not force:
        return {
            "ok": True,
            "skipped": True,
            "message": f"Database already has {existing} images; seeding skipped. Pass force=true to override.",
        }

    # Lazy import to avoid import-time DB setup costs
    from backend.scripts.import_images_from_json import import_images

    json_path = Path(__file__).resolve().parents[1] / "database" / "google_images_import.json"
    if not json_path.exists():
        raise HTTPException(status_code=500, detail=f"Seed file not found: {json_path}")

    result = import_images(str(json_path))
    return {"ok": True, "skipped": False, **result}
