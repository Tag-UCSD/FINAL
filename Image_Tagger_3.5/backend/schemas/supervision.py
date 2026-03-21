from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict


class TaggerPerformance(BaseModel):
    """Aggregate tagger stats for the Supervisor velocity view."""

    user_id: int
    username: str
    images_validated: int
    avg_duration_ms: int
    status: str = "active"

    model_config = ConfigDict(from_attributes=True)


class IRRStat(BaseModel):
    """Inter-rater reliability summary for a given image/attribute pair.

    This is intentionally simple and numerical so it can drive:
      * A heatmap in the Supervisor dashboard.
      * Drill-down into the Tag Inspector.
    """

    image_id: int
    filename: str
    attribute_key: str
    agreement_score: float
    conflict_count: int
    raters: List[str]

    model_config = ConfigDict(from_attributes=True)


class ValidationDetail(BaseModel):
    """Per-validation record for the Tag Inspector drawer."""

    id: int
    user_id: Optional[int] = None
    username: Optional[str] = None
    image_id: int
    attribute_key: str
    value: float
    duration_ms: Optional[int] = None
    created_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)