from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
import enum

class AlertConditionOperator(str, enum.Enum):
    GREATER_THAN = "GREATER_THAN"
    LESS_THAN = "LESS_THAN"
    EQUALS = "EQUALS"
    GREATER_THAN_OR_EQUAL_TO = "GREATER_THAN_OR_EQUAL_TO"
    LESS_THAN_OR_EQUAL_TO = "LESS_THAN_OR_EQUAL_TO"

class AlertConditionField(BaseModel):
    metric: str = Field(..., description="The metric to check, e.g., 'volatility.predicted', 'sentiment.score', 'price.close'. Use dot notation for nested fields.")
    operator: AlertConditionOperator = Field(..., description="The comparison operator.")
    value: float = Field(..., description="The value to compare against.")

class AlertConditions(BaseModel):
    logical_operator: str = Field("AND", description="Logical operator to combine conditions ('AND', 'OR').")
    conditions: List[AlertConditionField] = Field(..., description="List of specific conditions to evaluate.")
    description: Optional[str] = Field(None, description="User-friendly description of the alert conditions.")

class AlertBase(BaseModel):
    symbol: str = Field(..., example="AAPL", description="The financial symbol for the alert (e.g., AAPL, EURUSD).")
    conditions: AlertConditions = Field(..., description="Structured conditions for the alert to trigger.")
    notes: Optional[str] = Field(None, example="Watch for breakout", description="Optional user notes for the alert.")
    is_active: bool = Field(True, description="Whether the alert is currently active.")

class AlertCreate(AlertBase):
    pass

class AlertUpdate(BaseModel):
    symbol: Optional[str] = None
    conditions: Optional[AlertConditions] = None
    notes: Optional[str] = None
    is_active: Optional[bool] = None

class AlertResponse(AlertBase):
    id: int = Field(..., description="The unique identifier for the alert.")
    user_id: int = Field(..., description="The ID of the user who created the alert.")
    created_at: datetime = Field(..., description="The timestamp when the alert was created.")
    last_triggered_at: Optional[datetime] = Field(None, description="Timestamp of when the alert was last triggered.")

    class Config:
        from_attributes = True
