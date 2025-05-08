from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any
from datetime import date
from sqlalchemy.orm import Session
from ...services.economic_calendar import EconomicCalendarService
from ...services.auth import get_current_user
from ...database.models import User
from ...database.database import get_db
from ...schemas.event import EventResponse
import logging

router = APIRouter(prefix="/api/economic-calendar", tags=["economic-calendar"])

@router.get("/events", response_model=List[EventResponse])
async def get_economic_events(
    start_date: date = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end_date: date = Query(default=None, description="End date (YYYY-MM-DD)"),
    importance: str = Query(default="high", description="Minimum importance (low, medium, high, all)"),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get upcoming economic events."""
    try:
        service = EconomicCalendarService(db=db)
        events = await service.get_economic_events(start_date, end_date, importance)
        return events
    except Exception as e:
        logging.getLogger(__name__).error(f"Error getting economic events: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error fetching economic events.")
