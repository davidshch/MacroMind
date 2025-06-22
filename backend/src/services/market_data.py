"""Service for retrieving and processing basic market data."""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta
import asyncio
import pandas as pd
import numpy as np

from .market_data_services import AlphaVantageService, FinnhubService
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class MarketDataService:
    """Provides basic market data retrieval and processing."""

    def __init__(self):
        self.alpha_vantage = AlphaVantageService()
        self.finnhub = FinnhubService()
        self.cache = {}
        self.cache_duration = timedelta(minutes=15) # Cache duration for combined data
        self.vix_cache = None
        self.vix_cache_timestamp = None

    async def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get combined stock data including price, overview, and basic metrics.

        Args:
            symbol: The stock symbol (e.g., AAPL).

        Returns:
            A dictionary containing combined stock data.
        """
        cache_key = f"stock_data_{symbol}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        try:
            # Fetch data concurrently
            overview_task = self.alpha_vantage.get_overview(symbol)
            historical_task = self.alpha_vantage.get_historical_data(symbol, period="daily")
            metrics_task = self.finnhub.get_company_metrics(symbol)

            overview, historical_data, metrics = await asyncio.gather(
                overview_task, historical_task, metrics_task
            )

            if not historical_data:
                logger.warning(f"No historical data found for {symbol}, returning partial data.")
                # Return minimal data if history is missing
                return {
                    "symbol": symbol,
                    "overview": overview,
                    "metrics": metrics.get('metric', {}) if metrics else {},
                    "price_data": {"current_price": None, "previous_close": None, "change_percent": None},
                    "timestamp": datetime.now().isoformat()
                }

            # Process historical data for current price etc.
            latest_data = historical_data[0] # Assuming sorted descending by date
            previous_data = historical_data[1] if len(historical_data) > 1 else latest_data

            current_price = latest_data['close']
            previous_close = previous_data['close']
            change = current_price - previous_close
            change_percent = (change / previous_close) * 100 if previous_close else 0

            combined_data = {
                "symbol": symbol,
                "overview": overview,
                "metrics": metrics.get('metric', {}) if metrics else {},
                "price_data": {
                    "current_price": current_price,
                    "previous_close": previous_close,
                    "change": round(change, 2),
                    "change_percent": round(change_percent, 2),
                    "day_high": latest_data['high'],
                    "day_low": latest_data['low'],
                    "volume": latest_data['volume']
                },
                "timestamp": datetime.now().isoformat()
            }

            self._add_to_cache(cache_key, combined_data)
            return combined_data

        except Exception as e:
            logger.error(f"Error fetching combined stock data for {symbol}: {str(e)}")
            # Consider returning mock data or raising a specific exception
            # For now, raise HTTPException compatible error or return minimal structure
            raise Exception(f"Failed to retrieve data for {symbol}: {str(e)}") # Re-raise for API layer

    async def get_historical_prices(self, symbol: str, lookback_days: int = 30) -> List[Dict[str, Any]]:
        """Get historical price data for a symbol.

        Args:
            symbol: The stock symbol.
            lookback_days: Number of days to look back.

        Returns:
            A list of dictionaries containing date, open, high, low, close, volume.
            Returns data available up to the lookback period, might be less than lookback_days.
        """
        # Determine output size based on lookback_days
        output_size = "full" if lookback_days > 100 else "compact"

        try:
            # Use AlphaVantageService's method
            historical_data = await self.alpha_vantage.get_historical_data(symbol, period="daily")

            # AlphaVantageService returns data oldest -> newest. We want the MOST RECENT lookback_days.
            # Ensure chronological order first, then slice the tail.
            historical_data_sorted = sorted(
                [p for p in historical_data if p.get("date")],
                key=lambda p: p["date"]
            )

            recent_slice = historical_data_sorted[-lookback_days:]

            return recent_slice

        except Exception as e:
            logger.error(f"Error fetching historical prices for {symbol}: {str(e)}")
            return [] # Return empty list on error

    def get_vix_returns(self) -> pd.Series:
        """Get VIX returns data for correlation calculations.
        For testing, returns mock data if no real data is available."""
        if self.vix_cache is not None:
            if datetime.now() - self.vix_cache_timestamp < self.cache_duration:
                return self.vix_cache
            
        try:
            # In a real implementation, fetch actual VIX data
            # For now, return mock data
            dates = pd.date_range(end=datetime.now(), periods=252)  # One year of trading days
            vix_values = np.random.normal(20, 5, len(dates))  # Mock VIX values
            vix_series = pd.Series(vix_values, index=dates)
            returns = vix_series.pct_change()
            
            self.vix_cache = returns
            self.vix_cache_timestamp = datetime.now()
            
            return returns
        except Exception as e:
            logger.error(f"Error getting VIX returns: {str(e)}")
            # Return empty series as fallback
            return pd.Series()

    # --- Cache Helper Methods ---
    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get data from in-memory cache if still valid."""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.cache_duration:
                logger.debug(f"Cache hit for key: {key}")
                return data
            logger.debug(f"Cache expired for key: {key}")
            del self.cache[key] # Remove expired item
        logger.debug(f"Cache miss for key: {key}")
        return None

    def _add_to_cache(self, key: str, data: Dict[str, Any]):
        """Add data to in-memory cache with timestamp."""
        self.cache[key] = (data, datetime.now())
        logger.debug(f"Cached data for key: {key}")