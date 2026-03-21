from pydantic import BaseModel, ConfigDict
from typing import Optional

class ToolConfigUpdate(BaseModel):
    """Contract for modifying AI Model settings"""
    is_enabled: Optional[bool] = None
    cost_per_1k_tokens: Optional[float] = None
    
class ToolConfigRead(BaseModel):
    """Contract for reading AI Model state"""
    id: int
    name: str
    provider: str
    cost_per_1k_tokens: float
    is_enabled: bool
    
    model_config = ConfigDict(from_attributes=True)

class BudgetStatus(BaseModel):
    """Contract for Cost Dashboard"""
    total_spent: float
    hard_limit: float
    is_kill_switched: bool