# backend/models/__init__.py
# This file aggregates all models so Alembic can discover them for migrations.

from backend.database.core import Base
from backend.models.users import User
from backend.models.config import ToolConfig
from backend.models.assets import Image, Region
from backend.models.usage import ToolUsage
from backend.models.annotation import Validation

__all__ = ["Base", "User", "ToolConfig", "Image", "Region", "Validation"]
from backend.models.attribute import Attribute

__all__ = [
    "Base",
    "User",
    "ToolConfig",
    "Image",
    "Region",
    "Validation",
    "Attribute",
    "ToolUsage",
]

from backend.models.jobs import UploadJob, UploadJobItem

__all__ = list(dict.fromkeys(__all__ + ['UploadJob', 'UploadJobItem']))
