from sqlalchemy import String, Float, ForeignKey, Integer
from sqlalchemy.orm import Mapped, mapped_column, relationship
from backend.database.core import Base
from backend.database.mixins import TimestampMixin

class Validation(Base, TimestampMixin):
    """
    HITL Record.
    Records a human decision on a specific tag or region.
    Powers the 'Supervisor Dashboard' (IRR calculations).
    """
    __tablename__ = "validations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    image_id: Mapped[int] = mapped_column(ForeignKey("images.id"))
    
    # The attribute being validated (e.g., "spatial.prospect")
    attribute_key: Mapped[str] = mapped_column(String, index=True)
    
    # The value assigned (0.0 - 1.0 for continuous, or categorical)
    value: Mapped[float] = mapped_column(Float)
    
    # Optional: Link to a specific region if this is a local attribute
    region_id: Mapped[int] = mapped_column(Integer, nullable=True)

    # Velocity Tracking: How long did the user look at this before clicking?
    # Critical for detecting "spam clicking" by tired taggers
    duration_ms: Mapped[int] = mapped_column(Integer, default=0)
    
    user = relationship("User", back_populates="validations")
    image = relationship("Image", back_populates="validations")