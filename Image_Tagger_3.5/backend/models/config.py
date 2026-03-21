from sqlalchemy import String, Float, Boolean, JSON
from sqlalchemy.orm import Mapped, mapped_column
from backend.database.core import Base
from backend.database.mixins import TimestampMixin

class ToolConfig(Base, TimestampMixin):
    """
    System Configuration Table.
    Controls which AI models are active and tracks their pricing.
    This is the source of truth for the 'Cost & Governance Cockpit'.
    """
    __tablename__ = "tool_configs"

    id: Mapped[int] = mapped_column(primary_key=True)
    
    # e.g., "gpt-4v-preview", "sam-vit-h"
    name: Mapped[str] = mapped_column(String, unique=True, index=True)
    
    # e.g., "openai", "anthropic", "local"
    provider: Mapped[str] = mapped_column(String)
    
    # Cost Control
    # Used by CostEstimator service to calculate pre-run budget checks
    cost_per_1k_tokens: Mapped[float] = mapped_column(Float, default=0.0)
    cost_per_image: Mapped[float] = mapped_column(Float, default=0.0)
    
    # The "Kill Switch" - if False, the ModelLoader service will refuse to instantiate
    is_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    
    # Technical config (max tokens, temperature, local file paths)
    settings: Mapped[dict] = mapped_column(JSON, default={})