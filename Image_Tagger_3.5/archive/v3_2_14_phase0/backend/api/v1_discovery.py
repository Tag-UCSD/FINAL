from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
import json
import io
from fastapi.responses import StreamingResponse

from backend.database.core import get_db
from backend.services.auth import require_tagger\nfrom backend.services.training_export import TrainingExporter\nfrom backend.schemas.training import TrainingExample\n
from backend.services.query_builder import QueryBuilder
from backend.schemas.discovery import SearchQuery, ImageSearchResult, ExportRequest, AttributeRead
from backend.models.attribute import Attribute

router = APIRouter(prefix="/v1/explorer", tags=["Research Explorer"])

@router.post("/search", response_model=List[ImageSearchResult])
async def search_images(
    query: SearchQuery,
    db: Session = Depends(get_db),
    user = Depends(require_tagger),
):
    """
    High-level search endpoint for the Research Explorer.

    This currently delegates to QueryBuilder, which can be progressively
    enhanced without changing the API contract.
    """
    qb = QueryBuilder(db=db, user_id=user.id)
    results = qb.search_images(query)
    return results


@router.post("/export", response_model=list[TrainingExample])
async def export_dataset(
    request: ExportRequest,
    db: Session = Depends(get_db),
    user = Depends(require_tagger),
) -> list[TrainingExample]:
    """
    Export a slice of the validated dataset for fine-tuning / active learning.

    This is the Explorer-facing endpoint. It uses TrainingExporter to pull
    all Validation rows for the selected image_ids and returns them as a
    JSON list of TrainingExample records that the frontend can download.
    """
    exporter = TrainingExporter(db=db)
    examples = exporter.export_for_images(request.image_ids)
    return [TrainingExample(**e) for e in examples]


@router.get("/attributes", response_model=List[AttributeRead])
async def list_attributes(
    db: Session = Depends(get_db),
    user = Depends(require_tagger),
    category: str | None = None,
    limit: int = 500,
):
    """
    Return rows from the Attribute registry.

    This is the v3 counterpart of the v2 Feature Explorer's underlying
    taxonomy, now backed by the SQL model seeded from attributes.yml.
    """
    q = db.query(Attribute).filter(Attribute.is_active.is_(True))
    if category:
        q = q.filter(Attribute.category == category)
    q = q.order_by(Attribute.key).limit(limit)
    return q.all()