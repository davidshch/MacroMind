from pydantic import BaseModel
from datetime import datetime
from typing import Optional

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

    class Config:
        orm_mode = True
