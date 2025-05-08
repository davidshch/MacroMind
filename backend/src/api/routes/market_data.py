from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any, List
from sqlalchemy.orm import Session
import logging

from ...services.market_data import MarketDataService
from ...services.market_data_enhanced import EnhancedMarketDataService
from ...services.auth import get_current_user
from ...database.models import User
from ...database.database import get_db
from ...services.sector_fundamentals import SectorFundamentalsService, SECTOR_ETF_MAP # Added
from ...schemas.sector import SectorFundamentalResponse # Added
from datetime import date

router = APIRouter(prefix="/api/market", tags=["market"])
logger = logging.getLogger(__name__)

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

# --- Sector Fundamentals Endpoints ---

@router.get("/sectors/list", response_model=List[str])
async def list_supported_sectors(
    current_user: User = Depends(get_current_user)
):
    """List all supported sectors for fundamental analysis."""
    return list(SECTOR_ETF_MAP.keys())

@router.get("/sectors/{sector_name}/fundamentals", response_model=SectorFundamentalResponse)
async def get_sector_fundamental_data(
    sector_name: str,
    target_date: date = Depends(lambda: date.today()), # Default to today
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get fundamental data (P/E, P/B, Growth) for a specific sector."""
    try:
        service = SectorFundamentalsService(db=db)
        fundamentals = await service.get_sector_fundamentals(sector_name, target_date)
        if not fundamentals:
             # Service raises HTTPException, but double-check
             raise HTTPException(status_code=404, detail=f"Fundamentals not found for sector '{sector_name}' on {target_date}")
        return fundamentals
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.error(f"Error getting sector fundamentals for {sector_name}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error fetching sector fundamentals.")

@router.post("/sectors/update-all", response_model=List[SectorFundamentalResponse])
async def trigger_update_all_sector_fundamentals(
    current_user: User = Depends(get_current_user), # Add admin check later if needed
    db: Session = Depends(get_db)
):
    """Trigger an update for all sector fundamentals for the current day."""
    try:
        service = SectorFundamentalsService(db=db)
        updated_fundamentals = await service.update_all_sector_fundamentals(date.today())
        return updated_fundamentals
    except Exception as e:
        logger.error(f"Error triggering update for all sector fundamentals: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to trigger update for sector fundamentals.")