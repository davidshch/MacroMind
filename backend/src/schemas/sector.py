\
from pydantic import BaseModel
from typing import Optional
from datetime import date, datetime

class SectorFundamentalBase(BaseModel):
    sector_name: str
    date: date
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    earnings_growth: Optional[float] = None
    timestamp: datetime

class SectorFundamentalCreate(SectorFundamentalBase):
    pass

class SectorFundamentalResponse(SectorFundamentalBase):
    id: int

    class Config:
        from_attributes = True
