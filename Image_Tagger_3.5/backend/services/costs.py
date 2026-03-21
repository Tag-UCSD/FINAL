"""Cost and usage accounting helpers.

Sprint 1 goal:
- Provide a truthful, queryable ledger of VLM usage so the Admin
  budget dashboard no longer relies on hard-coded placeholder numbers.
- Keep the API surface small and dependency-free so it is easy to
  extend to other tools later.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from sqlalchemy import select, func

from backend.database.core import SessionLocal
from backend.models.usage import ToolUsage

logger = logging.getLogger(__name__)


def log_vlm_usage(
    provider: str,
    model_name: str,
    cost_usd: float,
    meta: Optional[Dict[str, Any]] = None,
) -> None:
    """Persist a single VLM usage record.

    This is intentionally fire-and-forget: any exception is logged and
    swallowed so that failures in the ledger never break the main
    request/response path.
    """
    db = SessionLocal()
    try:
        entry = ToolUsage(
            tool_name="vlm_analyze_image",
            provider=provider or "unknown",
            model_name=model_name or "unknown",
            cost_usd=float(cost_usd) if cost_usd is not None else 0.0,
            meta=meta or {},
        )
        db.add(entry)
        db.commit()
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to log VLM usage: %s", exc)
        db.rollback()
    finally:
        db.close()


def get_total_spent() -> float:
    """Return the total recorded spend in USD.

    If anything goes wrong (e.g. table missing on a fresh DB),
    we log and return 0.0 rather than breaking the Admin UI.
    """
    db = SessionLocal()
    try:
        stmt = select(func.coalesce(func.sum(ToolUsage.cost_usd), 0.0))
        result = db.execute(stmt).scalar_one()
        return float(result or 0.0)
    except Exception as exc:  # pragma: no cover - defensive
        logger.error("Failed to compute total spend: %s", exc)
        return 0.0
    finally:
        db.close()
