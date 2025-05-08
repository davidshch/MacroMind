from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect, Query
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from datetime import date

from ..services.ai_explanation import AIExplanationService, get_ai_explanation_service
from ..services.auth import get_current_user, get_current_active_user
from ..database.models import User
from ..database.database import get_db
from ..services.websocket import websocket_service_instance
from ..services.alerts import AlertService
from ..schemas.alert import AlertCreate, AlertResponse
from ..schemas.event import EventCreate, EventResponse
from ..services.economic_calendar import EconomicCalendarService, get_economic_calendar_service

# Dependency for AlertService
def get_alert_service(db: Session = Depends(get_db)) -> AlertService:
    return AlertService(db=db, websocket_service=websocket_service_instance)

router = APIRouter(prefix="/api", tags=["core", "vip", "websockets", "alerts", "calendar"])

# --- VIP Endpoint --- #
@router.get("/vip/explain/{symbol}", response_model=Dict[str, Any], tags=["vip"])
async def get_market_explanation(
    symbol: str,
    current_user: User = Depends(get_current_active_user),
    ai_service: AIExplanationService = Depends(get_ai_explanation_service)
):
    """(VIP) Get AI-powered market explanation for a symbol."""
    if not current_user.is_vip:
        raise HTTPException(status_code=403, detail="Access restricted to VIP users.")
    try:
        explanation = await ai_service.explain_market_conditions(symbol)
        return explanation
    except HTTPException as he:
        raise he
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.exception(f"Error in /vip/explain/{symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while generating the explanation for {symbol}.")

# --- WebSocket Endpoint --- #
@router.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """Establish WebSocket connection for a user."""
    await websocket_service_instance.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_service_instance.disconnect(websocket, user_id)
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"WebSocket error for user {user_id}: {e}")
        websocket_service_instance.disconnect(websocket, user_id)

# --- Alert Endpoints --- #
@router.post("/alerts", response_model=AlertResponse, tags=["alerts"])
async def create_alert_endpoint(
    alert: AlertCreate,
    current_user: User = Depends(get_current_active_user),
    alert_service: AlertService = Depends(get_alert_service)
):
    """Create a new alert for the current user."""
    return await alert_service.create_alert(user_id=current_user.id, alert_data=alert)

@router.get("/alerts", response_model=List[AlertResponse], tags=["alerts"])
async def get_user_alerts_endpoint(
    current_user: User = Depends(get_current_active_user),
    alert_service: AlertService = Depends(get_alert_service)
):
    """Get all active alerts for the current user."""
    return await alert_service.get_user_alerts(user_id=current_user.id)

@router.put("/alerts/{alert_id}", response_model=AlertResponse, tags=["alerts"])
async def update_alert_endpoint(
    alert_id: int,
    alert_update: AlertCreate,
    current_user: User = Depends(get_current_active_user),
    alert_service: AlertService = Depends(get_alert_service)
):
    """Update an existing alert for the current user."""
    updated_alert = await alert_service.update_alert(user_id=current_user.id, alert_id=alert_id, alert_update_data=alert_update)
    if updated_alert is None:
        raise HTTPException(status_code=404, detail="Alert not found or not owned by user")
    return updated_alert

@router.delete("/alerts/{alert_id}", status_code=204, tags=["alerts"])
async def delete_alert_endpoint(
    alert_id: int,
    current_user: User = Depends(get_current_active_user),
    alert_service: AlertService = Depends(get_alert_service)
):
    """Delete an alert for the current user."""
    success = await alert_service.delete_alert(user_id=current_user.id, alert_id=alert_id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found or not owned by user")
    return

# --- Economic Calendar Endpoints --- #
@router.post("/calendar/events", response_model=EventResponse, tags=["calendar"], status_code=201)
async def add_economic_event(
    event: EventCreate,
    current_user: User = Depends(get_current_active_user),
    calendar_service: EconomicCalendarService = Depends(get_economic_calendar_service)
):
    """Add a new economic event (requires authentication)."""
    return await calendar_service.add_event(event_data=event)

@router.get("/calendar/events", response_model=List[EventResponse], tags=["calendar"])
async def get_economic_events(
    start_date: date = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: date = Query(..., description="End date (YYYY-MM-DD)"),
    calendar_service: EconomicCalendarService = Depends(get_economic_calendar_service)
):
    """Get economic events within a date range."""
    if start_date > end_date:
        raise HTTPException(status_code=400, detail="Start date cannot be after end date")
    return await calendar_service.get_events_by_date_range(start_date=start_date, end_date=end_date)

@router.get("/calendar/events/upcoming", response_model=List[EventResponse], tags=["calendar"])
async def get_upcoming_economic_events(
    days: int = Query(7, ge=1, le=30, description="Number of days ahead to fetch events for"),
    calendar_service: EconomicCalendarService = Depends(get_economic_calendar_service)
):
    """Get upcoming economic events for the next N days."""
    return await calendar_service.get_upcoming_events(days_ahead=days)

@router.put("/calendar/events/{event_id}", response_model=EventResponse, tags=["calendar"])
async def update_economic_event(
    event_id: int,
    event_update: EventCreate,
    current_user: User = Depends(get_current_active_user),
    calendar_service: EconomicCalendarService = Depends(get_economic_calendar_service)
):
    """Update an existing economic event (requires authentication)."""
    updated_event = await calendar_service.update_event(event_id=event_id, event_update_data=event_update)
    if updated_event is None:
        raise HTTPException(status_code=404, detail=f"Event with ID {event_id} not found")
    return updated_event

@router.delete("/calendar/events/{event_id}", status_code=204, tags=["calendar"])
async def delete_economic_event(
    event_id: int,
    current_user: User = Depends(get_current_active_user),
    calendar_service: EconomicCalendarService = Depends(get_economic_calendar_service)
):
    """Delete an economic event (requires authentication)."""
    success = await calendar_service.delete_event(event_id=event_id)
    if not success:
        raise HTTPException(status_code=404, detail=f"Event with ID {event_id} not found")
    return
