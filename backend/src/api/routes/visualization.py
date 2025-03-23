from fastapi import APIRouter, HTTPException, Depends, Query
from typing import List, Dict, Any
from ...services.historical_data import HistoricalDataService
from ...services.auth import get_current_user
from ...database.models import User
from datetime import datetime

router = APIRouter(prefix="/api/visualization", tags=["visualization"])

@router.get("/historical-prices/{symbol}")
async def get_historical_prices(
    symbol: str,
    interval: str = Query(default="daily"),
    days: int = Query(default=30),
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get historical price data for charting."""
    try:
        service = HistoricalDataService()
        data = await service.get_historical_prices(symbol, interval)
        return data[-days:]  # Return only requested number of days
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sentiment-history/{symbol}")
async def get_sentiment_history(
    symbol: str,
    days: int = Query(default=30),
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get historical sentiment data."""
    try:
        service = HistoricalDataService()
        return await service.get_sentiment_history(symbol, days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/volatility-history/{symbol}")
async def get_volatility_history(
    symbol: str,
    days: int = Query(default=30),
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get historical volatility data."""
    try:
        service = HistoricalDataService()
        return await service.get_volatility_history(symbol, days)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/market-analysis/{symbol}")
async def get_market_analysis(
    symbol: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get comprehensive market analysis."""
    try:
        service = HistoricalDataService()
        return await service.get_market_analysis(symbol)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
