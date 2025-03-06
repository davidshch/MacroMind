from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from ...services.market_data import MarketDataService
from ...services.auth import get_current_user
from ...database.models import User
from ...database.database import get_db

router = APIRouter(prefix="/api/market-data", tags=["market-data"])

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