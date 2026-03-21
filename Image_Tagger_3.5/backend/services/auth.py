import os
from typing import Literal, Optional
from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel

Role = Literal["tagger", "scientist", "supervisor", "admin"]

# Security: Load Secret from ENV or default to a known dev key
# In production, this MUST be set via docker-compose
API_SECRET = os.getenv("API_SECRET", "dev_secret_key_change_me")

class CurrentUser(BaseModel):
    id: str
    role: Role

async def get_current_user(
    x_user_id: Optional[str] = Header(default=None, alias="X-User-Id"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_auth_token: Optional[str] = Header(default=None, alias="X-Auth-Token")
) -> CurrentUser:
    """
    Authenticated User Factory.
    
    Security Upgrade (v3.3):
    - Low privilege roles (tagger) can pass with just ID headers (internal trust).
    - High privilege roles (admin, supervisor) MUST provide the X-Auth-Token.
    """
    user_id = x_user_id or "1"
    role = (x_user_role or "tagger").lower()

    # Normalize role
    if role not in {"tagger", "scientist", "supervisor", "admin"}:
        role = "tagger"

    # RBAC Enforcement
    is_privileged = role in {"admin", "supervisor"}
    
    if is_privileged:
        if x_auth_token != API_SECRET:
            # Log this attempt in production
            print(f"SECURITY ALERT: Failed admin login attempt for user {user_id}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid Authentication Token for Privileged Role"
            )

    return CurrentUser(id=user_id, role=role)

def require_tagger(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    return user

def require_supervisor(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user.role not in {"supervisor", "admin"}:
        raise HTTPException(status_code=403, detail="Supervisor role required")
    return user

def require_admin(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user.role != "admin":
        raise HTTPException(status_code=403, detail="Admin role required")
    return user

def require_admin_or_supervisor(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if user.role not in {"admin", "supervisor"}:
        raise HTTPException(status_code=403, detail="Admin or Supervisor role required")
    return user
