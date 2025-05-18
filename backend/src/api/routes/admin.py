from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session
from ...database.database import get_db
from ...services.user import UserService
from ...services.auth import get_current_user
from ...services.volatility import VolatilityService
from ...schemas.volatility import TrainVolatilityModelRequest, TrainVolatilityModelResponse
from ...database.models import User
from ...config import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

router = APIRouter(prefix="/api/admin", tags=["admin"])

# Commenting out VIP-related admin logic for MVP submission
# class VIPUpdateRequest(BaseModel):
#     email: EmailStr
#     is_vip: bool

# Admin check based on email list in config
ADMIN_EMAILS = settings.admin_emails if hasattr(settings, 'admin_emails') and settings.admin_emails else ["david@gmail.com"]

async def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    if current_user.email not in ADMIN_EMAILS:
        raise HTTPException(
            status_code=403,
            detail="Admin access required"
        )
    return current_user

# @router.post("/users/vip", response_model=dict)
# async def update_vip_status(
#     request: VIPUpdateRequest,
#     db: Session = Depends(get_db),
#     admin: User = Depends(get_admin_user)
# ):
#     """Update user's VIP status (admin only)."""
#     try:
#         user_service = UserService(db)
#         user = user_service.update_vip_status(request.email, request.is_vip)
#         return {
#             "message": "VIP status updated successfully",
#             "email": user.email,
#             "is_vip": user.is_vip
#         }
#     except ValueError as e:
#         raise HTTPException(status_code=404, detail=str(e))
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

@router.post("/train-volatility-model", response_model=TrainVolatilityModelResponse)
async def train_volatility_model_endpoint(
    request: TrainVolatilityModelRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(get_admin_user),
    volatility_service: VolatilityService = Depends()
):
    """
    Admin endpoint to trigger volatility model training for a specific symbol.
    """
    logger.info(f"Admin request to train volatility model for symbol: {request.symbol}")
    try:
        training_result = await volatility_service.train_model_for_symbol_workflow(
            symbol=request.symbol,
            training_start_date_str=request.training_start_date_str,
            training_end_date_str=request.training_end_date_str,
            future_vol_period_days=request.future_vol_period_days
        )
        
        if training_result.get("status") == "error":
            logger.error(f"Volatility model training failed for {request.symbol}: {training_result.get('message')}")
            raise HTTPException(status_code=500, detail=training_result.get("message"))
            
        return TrainVolatilityModelResponse(**training_result)

    except HTTPException as http_exc:
        raise http_exc
    except ValueError as ve:
        logger.error(f"ValueError during model training for {request.symbol}: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Unexpected error training volatility model for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")
