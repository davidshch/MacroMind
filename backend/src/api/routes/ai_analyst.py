from fastapi import APIRouter, Depends
from ...schemas.ai_analyst import AIAnalystResponse
from ...services.ai_analyst import AIAnalystService, get_ai_analyst_service

router = APIRouter(tags=["AI Analyst"])

@router.get("/insights/{symbol}", response_model=AIAnalystResponse)
async def get_ai_analysis(
    symbol: str,
    service: AIAnalystService = Depends(get_ai_analyst_service),
):
    """
    Get AI-generated analysis and insights for a given symbol.
    """
    return await service.get_insights(symbol) 