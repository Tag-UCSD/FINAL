from typing import Literal, Optional

from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel


Role = Literal["tagger", "scientist", "supervisor", "admin"]


class CurrentUser(BaseModel):
    id: str
    role: Role


async def get_current_user(
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
) -> CurrentUser:
    """
    Minimal, header-based auth for v3.2 dev.

    This is intentionally simple so we can exercise RBAC and the
    different GUIs without a full identity provider. In production,
    replace with JWT / OAuth validation and a real User table lookup.
    """
    user_id = x_user_id or "1"
    role = (x_user_role or "tagger").lower()

    if role not in {"tagger", "scientist", "supervisor", "admin"}:
        # Default to least-privileged role if we get nonsense.
        role = "tagger"

    return CurrentUser(id=user_id, role=role)


def require_tagger(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Allow any authenticated role.

    Tagger, scientist, supervisor, and admin can all see Workbench/Explorer.
    """
    return user


def require_supervisor(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Require supervisor or admin role for monitoring endpoints."""
    if user.role not in {"supervisor", "admin"}:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Supervisor role required",
        )
    return user


def require_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    """Require admin role for configuration and kill-switch endpoints."""
    if user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin role required",
        )
    return user