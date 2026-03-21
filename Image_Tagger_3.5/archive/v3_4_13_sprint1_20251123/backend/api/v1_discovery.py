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

from typing import List

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from backend.database.session import get_db
from backend.services.auth import require_tagger
from backend.services.training_export import TrainingExporter, TrainingExample
from backend.schemas.discovery import SearchQuery, ImageSearchResult, ExportRequest, AttributeRead
from backend.models.attribute import Attribute

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

    limit = max(1, min(getattr(payload, "limit", 50), 200))
    offset = max(0, getattr(payload, "offset", 0))
    q = q.order_by(getattr(Image, "id")).offset(offset).limit(limit)

    images = q.all()
    results: List[ImageSearchResult] = []

    base_thumb_url = "/static/thumbnails"

    for img in images:
        image_id = getattr(img, "id", None)
        if image_id is None:
            continue
        thumb_name = getattr(img, "thumbnail_path", None) or f"image_{image_id}.jpg"
        thumb_url = f"{base_thumb_url}/{thumb_name}"

        # For now, we do not join attributes here; Explorer primarily needs
        # thumbnails and IDs to drive downstream detail views and exports.
        res = ImageSearchResult(image_id=image_id, thumbnail_url=thumb_url, attributes={})
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
    """
    attrs = db.query(Attribute).order_by(Attribute.key).all()
    results: List[AttributeRead] = []
    for attr in attrs:
        key = getattr(attr, "key", None)
        label = getattr(attr, "label", None) or key
        group = getattr(attr, "group", "default")
        if key is None:
            continue
        results.append(AttributeRead(key=key, label=label, group=group, description=getattr(attr, 'description', None) or getattr(attr, 'notes', None)))
    return results