from __future__ import annotations

from sqlalchemy import String, Float, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship

from backend.database.core import Base
from backend.database.mixins import TimestampMixin


class Validation(Base, TimestampMixin):
    """Human or pipeline validation record.

    This table is the central audit trail for:
      * Tagger Workbench decisions (human HITL).
      * Science pipeline auto-attributes (source = "science_pipeline_v3.3").
      * Supervisor Dashboard velocity and IRR calculations.

    Design notes
    ------------
    - ``user_id`` is nullable so that automated science jobs can emit
      validations without a concrete user.
    - ``attribute_key`` is enforced at the DB level to reference the
      canonical Attribute registry (``attributes.key``).
    - ``duration_ms`` provides the raw material for velocity / quality
      checks (e.g., detecting "spam clicking").
    """

    __tablename__ = "validations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(
        ForeignKey("users.id"),
        nullable=True,
        index=True,
    )
    image_id: Mapped[int] = mapped_column(
        ForeignKey("images.id"),
        index=True,
    )

    # The attribute being validated (e.g., "spatial.prospect").
    # In v3.4.63+ this is enforced at the DB level via a foreign key to
    # the canonical Attribute registry.
    attribute_key: Mapped[str] = mapped_column(
        String,
        ForeignKey("attributes.key"),
        index=True,
    )

    # The value assigned (0.0 - 1.0 for continuous, or categorical encoded
    # as a float for now).
    value: Mapped[float] = mapped_column(Float)

    # Optional: link to a specific region if this is a local attribute.
    region_id: Mapped[int | None] = mapped_column(Integer, nullable=True)

    # Origin of this validation: "manual", "science_pipeline_v3.3", etc.
    source: Mapped[str] = mapped_column(String, default="manual")

    # Velocity tracking: how long did the user look before clicking?
    # Critical for detecting "spam clicking" by tired taggers.
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)

    # Relationships
    user = relationship("User", back_populates="validations")
    image = relationship("Image", back_populates="validations")
    attribute = relationship("Attribute", back_populates="validations")
