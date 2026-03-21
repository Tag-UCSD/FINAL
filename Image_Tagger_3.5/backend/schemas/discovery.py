from pydantic import BaseModel, ConfigDict
from typing import List, Dict, Any, Optional
from datetime import datetime

class SearchQuery(BaseModel):
    """Contract for Complex Search"""
    query_string: str = ""
    filters: Dict[str, Any] = {}
    page: int = 1
    page_size: int = 20

class ImageSearchResult(BaseModel):
    """Contract for Masonry Grid Items - matches frontend expectations"""
    id: int
    url: str
    tags: List[str] = []
    meta_data: Dict[str, Any] = {}
    
class ExportRequest(BaseModel):
    """Contract for Dataset Export"""
    image_ids: List[int]
    format: str = "json"

class AttributeRead(BaseModel):
    """Attribute registry entry for Explorer filters."""
    id: int
    key: str
    name: str
    category: Optional[str] = None
    level: Optional[str] = None
    range: Optional[str] = None
    sources: Optional[str] = None
    notes: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class AttributeValue(BaseModel):
    """A single science-pipeline computed value for the detail view."""
    key: str
    name: str
    category: Optional[str] = None
    value: float
    source: str


class HumanValidation(BaseModel):
    """A single human-annotated validation record for the detail view."""
    user_id: Optional[int] = None
    username: Optional[str] = None
    attribute_key: str
    value: float
    duration_ms: Optional[int] = None
    created_at: Optional[datetime] = None


class TagInfo(BaseModel):
    """A single tag with provenance information for the detail view."""
    label: str
    source: str          # "preloaded" | "science_pipeline"
    source_label: str    # Human-readable: "Imported with dataset", "Semantic Tagger (VLM)", etc.
    confidence: Optional[float] = None   # 0.0–1.0, only for pipeline-derived tags
    attribute_key: Optional[str] = None  # e.g. "style.modern", only for pipeline-derived tags


class ImageDetailResult(BaseModel):
    """Full detail payload for the single-image viewer modal."""
    id: int
    url: str
    filename: str
    tags: List[TagInfo] = []
    meta_data: Dict[str, Any] = {}
    science_attributes: List[AttributeValue] = []
    human_validations: List[HumanValidation] = []