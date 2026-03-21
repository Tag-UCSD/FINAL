from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from typing import List
import io
from fastapi.responses import StreamingResponse

from backend.database.core import get_db
from backend.services.auth import require_tagger
from backend.services.training_export import TrainingExporter
from backend.services.query_builder import QueryBuilder
from backend.schemas.training import TrainingExample
from backend.schemas.discovery import SearchQuery, ImageSearchResult, ExportRequest, AttributeRead
from backend.models.attribute import Attribute
from backend.models.annotation import Validation


router = APIRouter(
    prefix="/v1/explorer",
    tags=["explorer"],
    dependencies=[Depends(require_tagger)],
)


@router.post("/search", response_model=ImageSearchResult)
def search_images(
    query: SearchQuery,
    db: Session = Depends(get_db),
):
    """
    Explorer GUI search endpoint.
    Aligns with QueryBuilder.execute(filters, page, page_size).
    """
    qb = QueryBuilder(db)
    # Expect SearchQuery.filters to be a dict of filter fields supported by QueryBuilder.
    results = qb.execute(
        filters=query.filters or {},
        page=query.page or 1,
        page_size=query.page_size or 24,
    )

    # Convert raw ORM Image rows to ImageSearchResult schema
    items = []
    for img in results.get("items", []):
        # Gather positive tags from validations
        try:
            vals = (
                db.query(Validation)
                .filter(Validation.image_id == img.id)
                .all()
            )
            tags = [
                v.attribute.name
                for v in vals
                if (v.value is not None and v.value > 0.5 and v.attribute is not None)
            ]
        except Exception:
            tags = []

        items.append(
            {
                "image_id": img.id,
                "url": getattr(img, "storage_path", "") or "",
                "tags": tags,
                "meta_data": getattr(img, "meta_data", None),
            }
        )

    return ImageSearchResult(
        items=items,
        total=results.get("total", len(items)),
        page=query.page or 1,
        page_size=query.page_size or 24,
    )


@router.post("/export")
def export_training_examples(
    req: ExportRequest,
    db: Session = Depends(get_db),
):
    exporter = TrainingExporter(db)
    stream = exporter.export_stream(req.format)
    filename = f"training_export.{req.format}"
    return StreamingResponse(
        io.BytesIO(stream.getvalue()),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@router.get("/attributes", response_model=List[AttributeRead])
def list_attributes(db: Session = Depends(get_db)):
    attrs = db.query(Attribute).order_by(Attribute.name.asc()).all()
    return [AttributeRead.model_validate(a) for a in attrs]