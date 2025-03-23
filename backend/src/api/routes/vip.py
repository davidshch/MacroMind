from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from ...services.ai_explanation import AIExplanationService
from ...services.auth import get_current_user
from ...database.models import User

router = APIRouter(prefix="/api/vip", tags=["vip"])

async def get_vip_user(current_user: User = Depends(get_current_user)) -> User:
    if not current_user.is_vip:
        raise HTTPException(
            status_code=403,
            detail="This endpoint requires VIP access"
        )
    return current_user

@router.get("/explain/{symbol}", response_model=Dict[str, Any])
async def get_market_explanation(
    symbol: str,
    current_user: User = Depends(get_vip_user)
):
    """Get AI-powered market explanation for VIP users."""
    try:
        service = AIExplanationService()
        explanation = await service.explain_market_conditions(symbol)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
