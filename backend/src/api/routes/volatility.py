"""Routes for volatility predictions and analysis."""

from fastapi import APIRouter, Depends, HTTPException, Query
from typing import List, Optional
from datetime import datetime, timedelta

from ...schemas.volatility import (
    VolatilityRequest,
    VolatilityResponse,
    HistoricalVolatility
)
from ...services.volatility import VolatilityService, get_volatility_service
from ...middleware.rate_limit import limiter
from ...services.auth import get_current_active_user
from ...database.models import User

router = APIRouter(prefix="/volatility", tags=["volatility"])

@router.post("/predict", response_model=VolatilityResponse)
@limiter.limit("30/minute")
async def predict_volatility(
    request: VolatilityRequest,
    volatility_service: VolatilityService = Depends(get_volatility_service),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get volatility predictions for a given symbol.
    
    Provides current volatility metrics, future predictions,
    and market regime analysis.
    """
    try:
        result = await volatility_service.calculate_and_predict_volatility(
            symbol=request.symbol,
            lookback_days=request.lookback_days,
            prediction_days=request.prediction_days
        )
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.exception(f"Error predicting volatility for {request.symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to predict volatility.")

@router.get("/historical/{symbol}", response_model=List[HistoricalVolatility])
@limiter.limit("20/minute")
async def get_historical_volatility(
    symbol: str,
    days: int = Query(90, ge=1, le=365),
    include_regime: bool = Query(True),
    volatility_service: VolatilityService = Depends(get_volatility_service),
    current_user: User = Depends(get_current_active_user)
):
    """Get historical volatility data with regime classification."""
    try:
        df = await volatility_service.get_historical_volatility(symbol, days)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data found")
        return df
    except HTTPException as he:
        raise he
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.exception(f"Error getting historical volatility for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve historical volatility.")

@router.get("/regime/{symbol}")
@limiter.limit("30/minute")
async def get_volatility_regime(
    symbol: str,
    volatility_service: VolatilityService = Depends(get_volatility_service),
    current_user: User = Depends(get_current_active_user)
):
    """Get current volatility regime and market conditions for a symbol."""
    try:
        result = await volatility_service.get_current_regime(symbol)
        return {
            "symbol": symbol,
            "timestamp": datetime.now(),
            **result
        }
    except HTTPException as he:
        raise he
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.exception(f"Error getting volatility regime for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve volatility regime.")
