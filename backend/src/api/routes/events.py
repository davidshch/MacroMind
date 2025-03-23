from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List
from ...database.database import get_db
from ...services.events import EventService
from ...schemas.event import EventCreate, EventResponse

router = APIRouter(prefix="/api/events", tags=["events"])

@router.get("/", response_model=List[EventResponse])
async def get_events(
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    events = EventService(db).get_events(skip=skip, limit=limit)
    return events

@router.get("/{event_id}", response_model=EventResponse)
async def get_event(event_id: int, db: Session = Depends(get_db)):
    event = EventService(db).get_event(event_id)
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    return event