from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any
from datetime import date
from sqlalchemy.orm import Session
from ...services.economic_calendar import EconomicCalendarService
from ...services.auth import get_current_user
from ...database.models import User
from ...database.database import get_db
from ...schemas.event import EventResponse, EventEchoResponse
from ...services.ml.model_factory import get_ml_model_factory
from ...services.market_data import get_market_data_service
import logging

router = APIRouter(prefix="/api/v1/calendar", tags=["Economic Calendar"])

# Dependency for EconomicCalendarService
async def get_economic_calendar_service(
    db: Session = Depends(get_db),
    ml_model_factory = Depends(get_ml_model_factory),
    market_data_service = Depends(get_market_data_service)
) -> EconomicCalendarService:
    return EconomicCalendarService(db=db, ml_model_factory=ml_model_factory, market_data_service=market_data_service)

@router.get("/events", response_model=List[EventResponse])
async def get_economic_events(
    start_date: date = Query(default=None, description="Start date (YYYY-MM-DD)"),
    end_date: date = Query(default=None, description="End date (YYYY-MM-DD)"),
    importance: str = Query(default="high", description="Minimum importance (low, medium, high, all)"),
    current_user: User = Depends(get_current_user),
    service: EconomicCalendarService = Depends(get_economic_calendar_service)
):
    """Get upcoming economic events, including AI-generated impact scores."""
    try:
        events = await service.get_economic_events(start_date, end_date, importance)
        return events
    except Exception as e:
        logging.getLogger(__name__).error(f"Error getting economic events: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error fetching economic events.")

@router.get(
    "/events/{event_id}/echo/{asset_symbol}", 
    response_model=EventEchoResponse,
    summary="Get Economic Event Echo Pattern",
    description="Analyzes historical price data for an asset around similar past economic events to show a typical pattern."
)
async def get_event_echo(
    event_id: int,
    asset_symbol: str,
    lookback_years: int = Query(5, ge=1, le=20, description="How many years of past events to consider."),
    window_pre_days: int = Query(5, ge=0, le=30, description="Number of trading days before event to include in pattern."),
    window_post_days: int = Query(10, ge=1, le=30, description="Number of trading days after event to include in pattern."),
    min_past_events: int = Query(3, ge=1, le=10, description="Minimum number of similar past events required."),
    service: EconomicCalendarService = Depends(get_economic_calendar_service),
    current_user: User = Depends(get_current_user)
):
    """Endpoint to get the economic event echo pattern for a specific event and asset."""
    try:
        echo_data = await service.get_event_echo_data(
            reference_event_id=event_id,
            asset_symbol=asset_symbol,
            lookback_years=lookback_years,
            window_pre_event_days=window_pre_days,
            window_post_event_days=window_post_days,
            min_past_events=min_past_events
        )
        return echo_data
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logging.getLogger(__name__).error(f"Error getting event echo for event {event_id}, symbol {asset_symbol}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error fetching event echo pattern.")
