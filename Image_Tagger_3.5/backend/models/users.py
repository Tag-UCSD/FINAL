from sqlalchemy import String, Boolean
from sqlalchemy.orm import Mapped, mapped_column, relationship
from typing import List
from backend.database.core import Base
from backend.database.mixins import TimestampMixin

class User(Base, TimestampMixin):
    """
    RBAC User Model.
    Roles:
      - 'admin': Full access, Cost Cockpit access.
      - 'scientist': Research Explorer access, Export access.
      - 'tagger': Tagger Workbench access only.
    """
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True)
    username: Mapped[str] = mapped_column(String, unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String)
    full_name: Mapped[str] = mapped_column(String, nullable=True)
    
    # Role Based Access Control
    role: Mapped[str] = mapped_column(String, default="tagger")
    
    # Soft delete / Access revocation
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)

    # Relationships
    # Using string reference "Validation" to avoid circular imports
    validations = relationship("Validation", back_populates="user")