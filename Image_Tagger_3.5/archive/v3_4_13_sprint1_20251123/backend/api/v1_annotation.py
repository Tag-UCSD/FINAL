from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Any

from backend.database.core import get_db
from backend.services.annotation import AnnotationService
from backend.services.auth import require_tagger
from backend.schemas.annotation import (
    ValidationRequest, 
    ValidationResponse, 
    RegionCreateRequest, 
    ImageWorkItem
)

# Router Definition
router = APIRouter(
    prefix="/v1/workbench",
    tags=["Tagger Workbench"],
    responses={404: {"description": "Not found"}},
)

@router.get("/next", response_model=ImageWorkItem)
async def get_next_image(
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_tagger)
):
    """
    WORKBENCH CORE: Fetches the next image for the user to annotate.
    Uses the 'Least Validated' queue algorithm.
    """
    service = AnnotationService(db)
    image = service.get_next_image_for_user(current_user["id"])
    
    if not image:
        raise HTTPException(status_code=404, detail="No more images in queue")
    
    # Construct response
    # Note: In production, 'url' would be signed S3 link or static Nginx path
    return ImageWorkItem(
        id=image.id,
        filename=image.filename,
        url=f"/static/{image.storage_path}", 
        meta_data=image.meta_data,
        created_at=image.created_at
    )

@router.post("/validate", response_model=ValidationResponse)
async def submit_validation(
    payload: ValidationRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_tagger)
):
    """
    HITL: Submit a judgment on an attribute (e.g., "This is Modern: 0.9").
    """
    service = AnnotationService(db)
    result = service.create_validation(current_user["id"], payload)
    
    return ValidationResponse(
        id=result.id,
        created_at=result.created_at,
        status="success"
    )

@router.post("/region", status_code=201)
async def create_region(
    payload: RegionCreateRequest,
    db: Session = Depends(get_db),
    current_user: dict = Depends(require_tagger)
):
    """
    HITL: Submit a new manual segmentation (box/polygon).
    """
    service = AnnotationService(db)
    result = service.create_region(current_user["id"], payload)
    
    return {
        "id": result.id,
        "status": "created",
        "label": result.manual_label
    }