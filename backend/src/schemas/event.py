from pydantic import BaseModel, Field
from datetime import datetime, date
from typing import Optional, List

class EventBase(BaseModel):
    name: str
    date: datetime
    description: str
    impact: str
    forecast: Optional[float] = None
    previous: Optional[float] = None
    actual: Optional[float] = None

class EventCreate(EventBase):
    pass

class EventResponse(EventBase):
    id: int
    impact_score: Optional[str] = Field(None, description="AI-generated potential market impact score (Low, Medium, High)")
    impact_confidence: Optional[float] = Field(None, description="Confidence level for the AI-generated impact score (0.0 to 1.0)")

    class Config:
        from_attributes = True  # Updated from orm_mode

class EventEchoPatternPoint(BaseModel):
    days_offset: int # e.g., -5, -1, 0 (event day), 1, 5
    avg_price_change_percentage: Optional[float] = None
    # Could add other stats like median_price_change, std_dev, etc.

class EventEchoResponse(BaseModel):
    event_name: str
    asset_symbol: str
    reference_event_date: date
    pattern_description: str # e.g., "Average price movement 5 days before and after similar events"
    echo_pattern: List[EventEchoPatternPoint]
    historical_events_analyzed: int
    message: Optional[str] = None
