from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession

from ...services.visualization import VisualizationService
from ...services.auth import get_current_user
from ...database.models import User
from ...database.database import get_db
from ...services.market_data import MarketDataService
from ...core.dependencies import get_market_data_service

router = APIRouter(tags=["visualization"])

# Dependency provider for VisualizationService
def get_visualization_service(
    db: AsyncSession = Depends(get_db),
    market_data_service: MarketDataService = Depends(get_market_data_service)
) -> VisualizationService:
    return VisualizationService(db, market_data_service)

@router.get("/historical-prices/{symbol}", response_model=List[Dict[str, Any]])
async def get_historical_prices(
    symbol: str,
    days: int = Query(default=90, ge=7, le=365),
    service: VisualizationService = Depends(get_visualization_service)
):
    """Get historical price data for charting."""
    try:
        data = await service.get_historical_prices(symbol, days)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sentiment-history/{symbol}", response_model=List[Dict[str, Any]])
async def get_sentiment_history(
    symbol: str,
    days: int = Query(default=90, ge=7, le=365),
    service: VisualizationService = Depends(get_visualization_service)
):
    """Get historical sentiment data for charting."""
    try:
        return await service.get_sentiment_history(symbol, days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Removing the other non-functional endpoints for now to keep it clean.
# @router.get("/volatility-history/{symbol}")
# ...
# @router.get("/market-analysis/{symbol}")
# ...
