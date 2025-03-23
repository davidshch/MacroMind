from fastapi import APIRouter, HTTPException, Depends, status
from typing import Dict, Any
from ...services.volatility import VolatilityService
from ...services.auth import get_current_user
from ...database.models import User
import logging

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/volatility", tags=["volatility"])

@router.get("/{symbol}", response_model=Dict[str, Any])
async def get_volatility(
    symbol: str,
    current_user: User = Depends(get_current_user)
):
    try:
        logger.info(f"Calculating volatility for {symbol}, requested by {current_user.email}")
        service = VolatilityService()
        volatility = await service.calculate_volatility(symbol)
        return volatility
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error calculating volatility: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to calculate volatility: {str(e)}"
        )
