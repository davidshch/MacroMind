from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any
from datetime import datetime, timedelta
from ...services.economic_calendar import EconomicCalendarService
from ...services.auth import get_current_user
from ...database.models import User

router = APIRouter(prefix="/api/economic-calendar", tags=["economic-calendar"])

@router.get("/events", response_model=List[Dict[str, Any]])
async def get_economic_events(
    start_date: datetime = Query(default=None),
    end_date: datetime = Query(default=None),
    importance: str = Query(default="high"),
    current_user: User = Depends(get_current_user)
):
    """Get upcoming economic events."""
    try:
        service = EconomicCalendarService()
        events = await service.get_economic_events(start_date, end_date, importance)
        return events
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/impact/{event_type}", response_model=Dict[str, Any])
async def get_event_impact(
    event_type: str,
    current_user: User = Depends(get_current_user)
):
    """Predict the impact of an economic event."""
    try:
        service = EconomicCalendarService()
        impact = await service.predict_event_impact(event_type)
        return impact
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
