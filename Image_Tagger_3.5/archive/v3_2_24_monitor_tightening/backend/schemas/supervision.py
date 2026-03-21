from pydantic import BaseModel
from datetime import datetime, ConfigDict
from typing import List

class TaggerPerformance(BaseModel):
    """Contract for Tagger Velocity Charts"""
    user_id: int
    username: str
    images_validated: int
    avg_duration_ms: int
    status: str = "active"
    
    model_config = ConfigDict(from_attributes=True)

class IRRStat(BaseModel):
    """Contract for Inter-Rater Reliability Heatmap"""
    image_id: int
    filename: str
    agreement_score: float
    conflict_count: int
    raters: List[str]

from datetime import datetime

class ValidationDetail(BaseModel):
    id: int
    user_id: int
    username: str
    attribute_key: str
    value: float
    duration_ms: int | None = None
    created_at: datetime | None = None