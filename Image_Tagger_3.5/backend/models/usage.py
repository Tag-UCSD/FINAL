from sqlalchemy import String, Float, JSON
from sqlalchemy.orm import Mapped, mapped_column

from backend.database.core import Base
from backend.database.mixins import TimestampMixin


class ToolUsage(Base, TimestampMixin):
    """Ledger of external tool usage (e.g., VLM calls).

    This table is intentionally generic so future tools can also log here.
    For Sprint 1, it is used primarily to track VLM image analyses and
    provide a truthful cost aggregate to the Admin budget dashboard.
    """

    __tablename__ = "tool_usage"

    id: Mapped[int] = mapped_column(primary_key=True)

    # High‑level tool identifier, e.g. "vlm_analyze_image"
    tool_name: Mapped[str] = mapped_column(String, index=True)

    # Provider and model are kept separate so we can aggregate by either.
    provider: Mapped[str] = mapped_column(String, index=True)
    model_name: Mapped[str] = mapped_column(String, index=True)

    # Estimated cost for this call, in USD.
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0)

    # Optional JSON metadata (e.g. raw usage, batch size, notes).
    meta: Mapped[dict] = mapped_column(JSON, default={})
