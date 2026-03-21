from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import DateTime, func
from datetime import datetime

class TimestampMixin:
    """
    Standard Audit Mixin.
    Ensures every table in the system tracks creation and modification times
    crucial for the 'Supervisor Dashboard' velocity tracking.
    """
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now(), 
        nullable=False
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        onupdate=func.now(), 
        nullable=True
    )