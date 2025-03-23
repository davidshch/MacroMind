from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from ...database.database import get_db
from ...services.user import UserService
from ...services.auth import get_current_user
from ...database.models import User

router = APIRouter(prefix="/api/admin", tags=["admin"])

class VIPUpdateRequest(BaseModel):
    email: EmailStr
    is_vip: bool

# For now, a simple admin check. In production, you'd want proper role-based access control
ADMIN_EMAILS = ["admin@example.com"]  # Add your admin email here

async def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.email not in ADMIN_EMAILS:
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    return current_user

@router.post("/users/vip", response_model=dict)
async def update_vip_status(
    request: VIPUpdateRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user)
):
    """Update user's VIP status (admin only)."""
    try:
        user_service = UserService(db)
        user = user_service.update_vip_status(request.email, request.is_vip)
        return {
            "message": "VIP status updated successfully",
            "email": user.email,
            "is_vip": user.is_vip
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
