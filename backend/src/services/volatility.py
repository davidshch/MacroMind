"""Volatility analysis and forecasting service."""

import numpy as np
from typing import Dict, Any, List
import aiohttp
from datetime import datetime, timedelta
from ..config import get_settings
from fastapi import HTTPException
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

class VolatilityService:
    """Market volatility analysis with trend prediction."""

    def __init__(self):
        """Initialize service with API and cache settings."""
        self.api_key = settings.api_key_alpha_vantage
        self.cache = {}
        self.cache_duration = timedelta(hours=24)

    async def calculate_volatility(self, symbol: str) -> Dict[str, Any]:
        """Calculate volatility metrics and trend."""
        try:
            historical_data = await self._fetch_historical_data(symbol)
            prices = [float(x['4. close']) for x in historical_data]
            daily_returns = np.diff(np.log(prices))
            
            historical_vol = np.std(daily_returns) * np.sqrt(252)
            recent_vol = np.std(daily_returns[-10:]) * np.sqrt(252)
            
            return {
                "symbol": symbol,
                "historical_volatility": float(historical_vol),
                "predicted_volatility": float(max(0, historical_vol + (recent_vol - historical_vol))),
                "market_conditions": self._assess_market_conditions(historical_vol, recent_vol),
                "last_updated": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Volatility calculation error: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to calculate volatility")

    async def _fetch_historical_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Fetch historical price data from Alpha Vantage."""
        url = "https://www.alphavantage.co/query"
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": "compact"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                if "Time Series (Daily)" not in data:
                    raise ValueError(f"No historical data found for {symbol}")
                return list(data["Time Series (Daily)"].values())

    def _assess_market_conditions(self, historical_vol: float, recent_vol: float) -> str:
        """Assess market conditions based on volatility patterns."""
        if recent_vol > historical_vol * 1.5:
            return "highly_volatile"
        elif recent_vol > historical_vol * 1.2:
            return "moderately_volatile"
        elif recent_vol < historical_vol * 0.8:
            return "low_volatility"
        else:
            return "normal"
