from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
from ...services.market_data import MarketDataService
from ...services.market_data_enhanced import EnhancedMarketDataService
from ...services.auth import get_current_user
from ...database.models import User
from ...database.database import get_db

router = APIRouter(prefix="/api/market", tags=["market"])

@router.get("/stock/{symbol}", response_model=Dict[str, Any])
async def get_stock_data(
    symbol: str,
    current_user: User = Depends(get_current_user)
):
    try:
        market_service = MarketDataService()
        data = await market_service.get_stock_data(symbol)
        return data
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/profile/{symbol}")
async def get_company_profile(
    symbol: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get detailed company profile."""
    service = EnhancedMarketDataService()
    return await service.get_company_profile(symbol)

@router.get("/metrics/{symbol}")
async def get_company_metrics(
    symbol: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get company financial metrics."""
    service = EnhancedMarketDataService()
    return await service.get_company_metrics(symbol)

@router.get("/earnings/{symbol}")
async def get_earnings_calendar(
    symbol: str,
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get upcoming earnings."""
    service = EnhancedMarketDataService()
    return await service.get_earnings_calendar(symbol)

@router.get("/price-target/{symbol}")
async def get_price_target(
    symbol: str,
    current_user: User = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get analyst price targets."""
    service = EnhancedMarketDataService()
    return await service.get_price_target(symbol)

@router.get("/news/{symbol}")
async def get_company_news(
    symbol: str,
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get recent company news."""
    service = EnhancedMarketDataService()
    return await service.get_company_news(symbol)

@router.get("/recommendations/{symbol}")
async def get_recommendation_trends(
    symbol: str,
    current_user: User = Depends(get_current_user)
) -> List[Dict[str, Any]]:
    """Get analyst recommendations."""
    service = EnhancedMarketDataService()
    return await service.get_recommendation_trends(symbol)

@router.get("/peers/{symbol}")
async def get_peer_companies(
    symbol: str,
    current_user: User = Depends(get_current_user)
) -> List[str]:
    """Get peer companies."""
    service = EnhancedMarketDataService()
    return await service.get_peer_companies(symbol)