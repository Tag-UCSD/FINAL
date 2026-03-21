from sqlalchemy import String, Boolean, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import Optional

from backend.database.core import Base


class Attribute(Base):
    """
    Attribute taxonomy entry.

    Ported from v2.6.3 contracts/attributes.yml into a proper SQLAlchemy model
    so downstream tools (Explorer, Tag Inspector, science pipeline) can reason
    over a shared attribute registry.
    """

    __tablename__ = "attributes"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    key: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    category: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    level: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    range: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    sources: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    source_version: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Relationships
    validations = relationship("Validation", back_populates="attribute")
