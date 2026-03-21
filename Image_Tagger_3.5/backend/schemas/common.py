from pydantic import BaseModel, ConfigDict
from datetime import datetime
from typing import Optional

class TimestampSchema(BaseModel):
    """Standard response mixin for timestamps"""
    created_at: datetime
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)