"""Routes for volatility predictions and analysis."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from scipy.stats import percentileofscore

from ...schemas.volatility import (
    VolatilityRequest,
    VolatilityResponse,
    HistoricalVolatility
)
from ...services.volatility import VolatilityService, get_volatility_service
from ...middleware.rate_limit import limiter
from ...services.auth import get_current_active_user
from ...database.models import User

import logging
logger = logging.getLogger(__name__)

router = APIRouter(tags=["volatility"])

@router.post("/predict", response_model=VolatilityResponse)
@limiter.limit("30/minute")
async def predict_volatility(
    request: Request,
    volatility_request: VolatilityRequest,
    volatility_service: VolatilityService = Depends(get_volatility_service),
    current_user: User = Depends(get_current_active_user)
):
    """
    Get volatility predictions for a given symbol.
    
    Provides current volatility metrics, future predictions,
    and market regime analysis.
    """
    logger = logging.getLogger(__name__)
    logger.debug(f"Received volatility prediction request for {volatility_request.symbol}")
    
    try:
        logger.debug(f"Calling volatility service with params: symbol={volatility_request.symbol}, "
                    f"lookback={volatility_request.lookback_days}, "
                    f"prediction_days={volatility_request.prediction_days}")
        
        result = await volatility_service.calculate_and_predict_volatility(
            symbol=volatility_request.symbol,  
            lookback_days=volatility_request.lookback_days,  
            prediction_days=volatility_request.prediction_days  
        )
        logger.debug(f"Successfully calculated volatility for {volatility_request.symbol}")
        return result
    except HTTPException as he:
        logger.error(f"HTTP error in volatility prediction: {str(he)}")
        raise he
    except Exception as e:
        logger.exception(f"Error predicting volatility for {volatility_request.symbol}: {e}")  
        raise HTTPException(status_code=500, detail="Failed to predict volatility.")

@router.get("/historical/{symbol}", response_model=List[HistoricalVolatility])
@limiter.limit("20/minute")
async def get_historical_volatility(
    request: Request,
    symbol: str,
    days: int = 90,
    volatility_service: VolatilityService = Depends(get_volatility_service),
    current_user: User = Depends(get_current_active_user)
) -> List[Dict[str, Any]]:
    """Get historical volatility data with regime classifications."""
    try:
        features_df = await volatility_service._prepare_comprehensive_features_df(
            symbol,
            datetime.now(),
            data_fetch_lookback_days=days
        )

        if features_df.empty:
            logger.warning(f"No historical data found for {symbol}")
            return []

        result = []
        for idx, row in features_df.iterrows():
            # Filter the DataFrame up to the current index
            vol_series = features_df.loc[:idx, 'historical_vol_10d']
            regime = volatility_service._detect_volatility_regime(features_df.loc[:idx])

            vol_value = row['historical_vol_10d']
            percentile = float(percentileofscore(vol_series, vol_value) if len(vol_series) > 1 else 50.0)

            result.append({
                "date": idx.isoformat(),
                "value": float(vol_value),
                "regime": regime,
                "percentile": percentile
            })

        return result

    except Exception as e:
        logger.exception(f"Error getting historical volatility for {symbol}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve historical volatility: {str(e)}"
        )

@router.get("/regime/{symbol}")
@limiter.limit("30/minute")
async def get_volatility_regime(
    request: Request,
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
