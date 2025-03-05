from backend.database import SessionLocal, EconomicEvent
from sqlalchemy.orm import Session

def save_economic_events(events):
    db: Session = SessionLocal()
    for event in events:
        db_event = EconomicEvent(
            event_name=event["event_name"],
            date=event["date"],
            impact=event["impact"]
        )
        db.add(db_event)
    db.commit()
    db.close()