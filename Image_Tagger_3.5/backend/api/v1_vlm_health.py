"""VLM health and Turing-test API endpoints.

This router exposes a minimal read-only view over the VLM health reports
produced by the CLI/SOP in docs/ops/VLM_Health_SOP.md.

It allows the Admin frontend to:
- List available health runs under reports/vlm_health/
- Download the variance-audit CSV for a run
- Download the Turing summary text file for a run
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

from backend.services.auth import require_admin

router = APIRouter(prefix="/v1/vlm-health", tags=["vlm-health"])

# Root directory where CLI scripts write health reports.
# Layout:
#   reports/vlm_health/
#       RUN_ID/
#           raw/
#           derived/
#           log.md
VLM_HEALTH_ROOT = Path(os.getenv("VLM_HEALTH_ROOT", "reports/vlm_health")).resolve()


class VLMHealthRun(BaseModel):
    """Summary metadata for a single VLM health run."""

    run_id: str
    created_at: Optional[datetime]
    has_variance_audit: bool
    has_turing_summary: bool


def _get_runs_root() -> Path:
    return VLM_HEALTH_ROOT


def _safe_file(run_id: str, relative: str) -> Path:
    """Resolve a file inside a run folder and guard against traversal."""
    root = _get_runs_root()
    run_dir = (root / run_id).resolve()
    root_resolved = root.resolve()

    if not str(run_dir).startswith(str(root_resolved)):
        raise HTTPException(status_code=400, detail="Invalid run identifier")

    target = (run_dir / relative).resolve()
    if not str(target).startswith(str(run_dir)):
        raise HTTPException(status_code=400, detail="Invalid file path")

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    return target


@router.get("/runs", response_model=List[VLMHealthRun])
def list_runs(user = Depends(require_admin)) -> List[VLMHealthRun]:
    """List all known VLM health runs, newest first.

    This inspects the reports/vlm_health directory and surfaces a small
    metadata record per RUN_ID. It does not read any of the large CSV
    contents; the Admin UI can fetch those lazily on demand.
    """
    root = _get_runs_root()
    if not root.exists() or not root.is_dir():
        return []

    items: List[VLMHealthRun] = []

    for entry in root.iterdir():
        if not entry.is_dir():
            continue

        run_id = entry.name
        derived_dir = entry / "derived"
        variance_path = derived_dir / "vlm_variance_audit.csv"
        summary_path = derived_dir / "vlm_turing_summary.txt"

        try:
            stat = entry.stat()
            created_at: Optional[datetime] = datetime.fromtimestamp(stat.st_mtime)
        except Exception:
            created_at = None

        items.append(
            VLMHealthRun(
                run_id=run_id,
                created_at=created_at,
                has_variance_audit=variance_path.exists(),
                has_turing_summary=summary_path.exists(),
            )
        )

    # Sort newest first by created_at (fallback to name)
    items.sort(
        key=lambda r: (
            r.created_at or datetime.min,
            r.run_id,
        ),
        reverse=True,
    )
    return items


@router.get("/runs/{run_id}/variance-audit")
def download_variance_audit(run_id: str, user = Depends(require_admin)):
    """Download the variance audit CSV for a given run."""
    path = _safe_file(run_id, "derived/vlm_variance_audit.csv")
    filename = f"{run_id}_vlm_variance_audit.csv"
    return FileResponse(
        path,
        media_type="text/csv",
        filename=filename,
    )


@router.get("/runs/{run_id}/turing-summary")
def download_turing_summary(run_id: str, user = Depends(require_admin)):
    """Download the Turing-test summary text file for a given run."""
    path = _safe_file(run_id, "derived/vlm_turing_summary.txt")
    filename = f"{run_id}_vlm_turing_summary.txt"
    return FileResponse(
        path,
        media_type="text/plain",
        filename=filename,
    )
