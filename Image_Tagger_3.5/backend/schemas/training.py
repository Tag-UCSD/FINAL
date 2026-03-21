from datetime import datetime
from pydantic import BaseModel

class TrainingExample(BaseModel):
    """Single row of exported training data.

    This mirrors the core fields produced by TrainingExporter and can be
    consumed by downstream tools or written to JSON/JSONL.
    """
    image_id: int
    image_filename: str
    attribute_key: str
    value: float
    user_id: int
    region_id: int | None = None
    duration_ms: int | None = None
    created_at: datetime | None = None
    source: str | None = None