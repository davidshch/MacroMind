from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from ...services.ai_explanation import AIExplanationService
from ...services.auth import get_current_user
from ...database.models import User

router = APIRouter(prefix="/api/vip", tags=["vip"])

@router.get("/explain/{symbol}", response_model=Dict[str, Any])
async def get_market_explanation(
    symbol: str,
    current_user: User = Depends(get_current_user)  # Only require normal authentication
):
    """Get AI-powered market explanation."""
    try:
        service = AIExplanationService()
        explanation = await service.explain_market_conditions(symbol)
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
