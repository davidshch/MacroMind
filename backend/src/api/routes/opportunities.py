"""
API routes for Market Opportunity Highlighter.
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, status

from src.schemas.opportunity import MarketOpportunityResponse
from src.services.opportunity_highlighter import OpportunityHighlightService
from src.services.volatility import VolatilityService, get_volatility_service # Dependency
from src.services.sentiment_analysis import SentimentAnalysisService, get_sentiment_service # Dependency
from src.database.models import User as UserModel
from src.services.auth import get_current_active_user

logger = logging.getLogger(__name__)
router = APIRouter(
    prefix="/api/v1/opportunities", 
    tags=["Market Opportunities"]
)

# Dependency for OpportunityHighlightService
async def get_opportunity_highlight_service(
    volatility_service: VolatilityService = Depends(get_volatility_service),
    sentiment_service: SentimentAnalysisService = Depends(get_sentiment_service)
) -> OpportunityHighlightService:
    return OpportunityHighlightService(
        volatility_service=volatility_service, 
        sentiment_service=sentiment_service
    )

@router.get(
    "/highlights", 
    response_model=MarketOpportunityResponse,
    summary="Get Market Opportunity Highlights",
    description="Identifies and returns assets with concurrent high forecasted volatility and strong directional sentiment."
)
async def get_market_opportunity_highlights(
    # Allow users to submit a list of symbols, otherwise use default popular list
    symbols: Optional[List[str]] = Query(None, description="Optional list of asset symbols to check. Defaults to a predefined popular list if not provided."),
    service: OpportunityHighlightService = Depends(get_opportunity_highlight_service),
    current_user: UserModel = Depends(get_current_active_user) # Ensure user is authenticated
):
    """
    Endpoint to get market opportunity highlights.
    """
    try:
        opportunity_data = await service.get_market_opportunities(asset_list=symbols)
        return opportunity_data
    except HTTPException as http_exc:
        logger.error(f"HTTPException during market opportunity highlighting: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.exception(f"Unexpected error during market opportunity highlighting: {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to retrieve market opportunity highlights.")
