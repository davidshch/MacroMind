from fastapi import APIRouter, HTTPException, Depends
from typing import List, Dict, Any
from ...services.alerts import AlertService
from ...services.auth import get_current_user
from ...database.models import User
from ...database.database import get_db
from sqlalchemy.orm import Session
from pydantic import BaseModel

router = APIRouter(prefix="/api/alerts", tags=["alerts"])

class AlertCreate(BaseModel):
    symbol: str
    alert_type: str
    condition: Dict[str, Any]

@router.post("/create", response_model=Dict[str, Any])
async def create_alert(
    alert: AlertCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Create a new alert."""
    try:
        service = AlertService(db)
        new_alert = await service.create_alert(
            current_user.id,
            alert.symbol,
            alert.alert_type,
            alert.condition
        )
        return new_alert
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/list", response_model=List[Dict[str, Any]])
async def list_alerts(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get all alerts for the current user."""
    service = AlertService(db)
    return await service.get_user_alerts(current_user.id)

@router.delete("/{alert_id}")
async def delete_alert(
    alert_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Delete an alert."""
    service = AlertService(db)
    success = await service.delete_alert(alert_id, current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Alert not found")
    return {"message": "Alert deleted successfully"}
