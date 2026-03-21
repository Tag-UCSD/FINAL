from pydantic import BaseModel
from typing import Dict, List, Optional
from pathlib import Path
from fastapi import UploadFile, File,  APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import select, update

from backend.database.core import get_db
from backend.services.auth import require_admin
from backend.services.vlm import describe_vlm_configuration, update_vlm_config, get_vlm_engine, StubEngine
from backend.models.config import ToolConfig
from backend.schemas.admin import ToolConfigRead, ToolConfigUpdate, BudgetStatus
from backend.schemas.discovery import ExportRequest
from backend.schemas.training import TrainingExample
from backend.services.training_export import TrainingExporter


router = APIRouter(prefix="/v1/admin", tags=["Admin Cockpit"])


@router.get("/models", response_model=list[ToolConfigRead])
async def list_models(
    db: Session = Depends(get_db),
    user = Depends(require_admin),
) -> list[ToolConfigRead]:
    """Return all ToolConfig rows.

    This powers the 'AI Models' section of the Admin Cockpit.
    """
    tools = db.execute(select(ToolConfig)).scalars().all()
    return [ToolConfigRead.model_validate(t) for t in tools]


@router.patch("/models/{tool_id}", response_model=ToolConfigRead)
async def update_model(
    tool_id: int,
    payload: ToolConfigUpdate,
    db: Session = Depends(get_db),
    user = Depends(require_admin),
) -> ToolConfigRead:
    """Update a single model's enabled state and/or cost.

    Frontend calls this when toggling a model or editing its cost.
    """
    tool = db.get(ToolConfig, tool_id)
    if tool is None:
        raise HTTPException(status_code=404, detail="ToolConfig not found")

    if payload.is_enabled is not None:
        tool.is_enabled = payload.is_enabled
    if payload.cost_per_1k_tokens is not None:
        tool.cost_per_1k_tokens = float(payload.cost_per_1k_tokens)

    db.add(tool)
    db.commit()
    db.refresh(tool)
    return ToolConfigRead.model_validate(tool)


@router.get("/budget", response_model=BudgetStatus)
async def get_budget(
    db: Session = Depends(get_db),
    user = Depends(require_admin),
) -> BudgetStatus:
    """Return a simple cost snapshot and kill-switch state.

    In this v3.2 dev build, the numbers are conservative placeholders.
    The kill-switch is derived from whether any paid models (cost_per_1k_tokens > 0)
    remain enabled.
    """
    paid_enabled = db.execute(
        select(ToolConfig).where(
            ToolConfig.cost_per_1k_tokens > 0,
            ToolConfig.is_enabled.is_(True),
        ).limit(1)
    ).first()
    is_kill_switched = paid_enabled is None

    # TODO: integrate with a real CostEstimator / usage ledger.
    total_spent = 14.50
    hard_limit = 15.00

    return BudgetStatus(
        total_spent=total_spent,
        hard_limit=hard_limit,
        is_kill_switched=is_kill_switched,
    )


@router.post("/kill-switch", response_model=BudgetStatus)
async def kill_switch(
    active: bool,
    db: Session = Depends(get_db),
    user = Depends(require_admin),
) -> BudgetStatus:
    """Global kill switch for paid models.

    When activated, all ToolConfigs with cost_per_1k_tokens > 0 are disabled.
    We then return a fresh BudgetStatus so the UI can update seamlessly.
    """
    if active:
        stmt = (
            update(ToolConfig)
            .where(ToolConfig.cost_per_1k_tokens > 0)
            .values(is_enabled=False)
        )
        db.execute(stmt)
        db.commit()

    # Re-use the logic from get_budget to compute state
    paid_enabled = db.execute(
        select(ToolConfig).where(
            ToolConfig.cost_per_1k_tokens > 0,
            ToolConfig.is_enabled.is_(True),
        ).limit(1)
    ).first()
    is_kill_switched = paid_enabled is None

    total_spent = 14.50
    hard_limit = 15.00

    return BudgetStatus(
        total_spent=total_spent,
        hard_limit=hard_limit,
        is_kill_switched=is_kill_switched,
    )
@router.post("/training/export", response_model=list[TrainingExample])
async def admin_export_training(
    request: ExportRequest,
    db: Session = Depends(get_db),
    user = Depends(require_admin),
) -> list[TrainingExample]:
    """
    Admin-facing training export endpoint.

    This mirrors the Explorer export but is scoped to admin RBAC and
    can be extended with richer filters (quality thresholds, time
    windows) in future steps.
    """
    exporter = TrainingExporter(db=db)
    examples = exporter.export_for_images(request.image_ids)
    return [TrainingExample(**e) for e in examples]

@router.get("/export/images")
def export_all_images(
    db: Session = Depends(get_db),
    user = Depends(require_admin),
):
    """Download a zip of all stored image files.

    This is an admin-only convenience endpoint for researchers who
    need local copies of the raw assets. In typical classroom
    usage, datasets are modest in size, so an in-memory zip is
    acceptable.
    """
    import io as _io
    import os as _os
    import zipfile as _zip
    from pathlib import Path as _Path
    from backend.models.assets import Image  # type: ignore

    try:
        from backend.settings import IMAGE_STORAGE_ROOT  # type: ignore
    except Exception:
        IMAGE_STORAGE_ROOT = _os.getenv("IMAGE_STORAGE_ROOT", "data_store")

    images = db.query(Image).all()
    if not images:
        raise HTTPException(status_code=404, detail="No images available for export")

    buf = _io.BytesIO()
    root = _Path(IMAGE_STORAGE_ROOT)

    with _zip.ZipFile(buf, "w", _zip.ZIP_DEFLATED) as zf:
        for img in images:
            path = _Path(img.storage_path)
            if not path.is_absolute():
                path = root / path
            if not path.exists():
                # Skip missing files rather than failing completely.
                continue
            try:
                arcname = path.relative_to(root)
            except Exception:
                arcname = path.name
            zf.write(path, arcname=str(arcname))

    buf.seek(0)
    headers = {
        "Content-Disposition": 'attachment; filename="image_tagger_images_export.zip"'
    }
    return StreamingResponse(buf, media_type="application/zip", headers=headers)
class AdminUploadResult(BaseModel):
    created_count: int
    image_ids: List[int]
    storage_paths: List[str]


@router.post("/upload", response_model=AdminUploadResult)
async def upload_images(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    user=Depends(require_admin),
) -> AdminUploadResult:
    """Bulk image upload endpoint for the Admin Cockpit.

    This accepts one or more image files and stores them under the
    configured IMAGE_STORAGE_ROOT. For each stored file, a new Image
    row is created in the database.
    """
    import os as _os
    from pathlib import Path as _Path
    from uuid import uuid4

    try:
        from backend.settings import IMAGE_STORAGE_ROOT  # type: ignore
    except Exception:
        IMAGE_STORAGE_ROOT = _os.getenv("IMAGE_STORAGE_ROOT", "data_store")

    from backend.models.assets import Image  # type: ignore

    root = _Path(IMAGE_STORAGE_ROOT)
    root.mkdir(parents=True, exist_ok=True)

    created_ids: List[int] = []
    storage_paths: List[str] = []

    for f in files:
        suffix = "".join(_Path(f.filename or "uploaded").suffixes)
        if suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp"}:
            # Skip unknown formats; in future we might reject the request instead.
            continue
        unique_name = f"{uuid4().hex}{suffix}"
        dest = root / unique_name
        content = await f.read()
        dest.write_bytes(content)

        image = Image(storage_path=str(dest), source="admin_upload")
        db.add(image)
        db.flush()
        created_ids.append(image.id)
        storage_paths.append(str(dest))

    db.commit()

    return AdminUploadResult(
        created_count=len(created_ids),
        image_ids=created_ids,
        storage_paths=storage_paths,
    )



class VLMConfigRequest(BaseModel):
    provider: str = "auto"
    cognitive_prompt_override: Optional[str] = None
    max_batch_size: Optional[int] = None
    cost_per_1k_images_usd: Optional[float] = None


@router.get("/vlm/config")
def get_vlm_config(
    db: Session = Depends(get_db),
    user = Depends(require_admin),
) -> Dict[str, Any]:
    """Return the current VLM configuration and detected backends."""
    return describe_vlm_configuration()


@router.post("/vlm/config")
def set_vlm_config(
    payload: VLMConfigRequest,
    db: Session = Depends(get_db),
    user = Depends(require_admin),
) -> Dict[str, Any]:
    """Persist the chosen VLM settings (provider, prompt override, batch size, cost hint)."""
    return update_vlm_config(
        provider=payload.provider,
        cognitive_prompt_override=payload.cognitive_prompt_override,
        max_batch_size=payload.max_batch_size,
        cost_per_1k_images_usd=payload.cost_per_1k_images_usd,
    )


class VLMTestRequest(BaseModel):
    image_id: int
    prompt: str = "Describe this architectural image in one or two sentences."


@router.post("/vlm/test")
def test_vlm_configuration(
    payload: VLMTestRequest,
    db: Session = Depends(get_db),
    user = Depends(require_admin),
) -> Dict[str, Any]:
    """Test the currently configured VLM on a single stored image."""
    from backend.models.assets import Image

    img = db.query(Image).get(payload.image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    path = Path(img.storage_path)
    if not path.is_absolute():
        path = Path("data_store") / path
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"File not found at {path}")

    with open(path, "rb") as f:
        image_bytes = f.read()

    engine = get_vlm_engine()
    try:
        result = engine.analyze_image(image_bytes, payload.prompt)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"VLM error: {exc}")

    return {
        "status": "success",
        "engine": type(engine).__name__,
        "is_stub": isinstance(engine, StubEngine) or bool(result.get("stub")),
        "response": result,
    }
