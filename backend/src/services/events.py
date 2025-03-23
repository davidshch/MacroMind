"""Economic event tracking service."""

from sqlalchemy.orm import Session
from ..database.models import EconomicEvent
from typing import List, Optional

class EventService:
    """Manages economic calendar events."""
    
    def __init__(self, db: Session):
        self.db = db

    def get_events(self, skip: int = 0, limit: int = 100) -> List[EconomicEvent]:
        """Get paginated economic events."""
        return self.db.query(EconomicEvent)\
            .offset(skip)\
            .limit(limit)\
            .all()

    def get_event(self, event_id: int) -> Optional[EconomicEvent]:
        """Get single event by ID."""
        return self.db.query(EconomicEvent)\
            .filter(EconomicEvent.id == event_id)\
            .first()

    def create_event(self, event_data: dict) -> EconomicEvent:
        """Create new economic event."""
        event = EconomicEvent(**event_data)
        self.db.add(event)
        self.db.commit()
        self.db.refresh(event)
        return event

    def update_event(self, event_id: int, event_data: dict) -> Optional[EconomicEvent]:
        """Update existing event."""
        event = self.get_event(event_id)
        if event:
            for key, value in event_data.items():
                setattr(event, key, value)
            self.db.commit()
            self.db.refresh(event)
        return event

    def delete_event(self, event_id: int) -> bool:
        """Delete event by ID."""
        event = self.get_event(event_id)
        if event:
            self.db.delete(event)
            self.db.commit()
            return True
        return False
