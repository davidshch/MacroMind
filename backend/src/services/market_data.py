"""Service for fetching real-time stock market data from Alpha Vantage."""

import aiohttp
from typing import Dict, Any, Optional
from ..config import get_settings
from fastapi import HTTPException
from datetime import datetime, timedelta

settings = get_settings()

class MarketDataService:
    """Basic market data service with caching."""
    
    def __init__(self):
        self.api_key = settings.api_key_alpha_vantage
        self.base_url = "https://www.alphavantage.co/query"
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)

    async def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """
        Get current market data for a stock.

        Args:
            symbol: Stock ticker symbol

        Returns:
            dict: Current price, volume and change data

        Raises:
            HTTPException: If symbol not found or API error
        """
        cache_key = f"stock_{symbol}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        params = {
            "function": "GLOBAL_QUOTE",
            "symbol": symbol,
            "apikey": self.api_key
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url, params=params) as response:
                if response.status != 200:
                    raise HTTPException(status_code=response.status, detail="Failed to fetch market data")
                
                data = await response.json()
                if "Global Quote" not in data:
                    raise HTTPException(status_code=404, detail="Symbol not found")
                
                processed_data = self._process_stock_data(data["Global Quote"])
                self._add_to_cache(cache_key, processed_data)
                return processed_data

    def _process_stock_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Format raw API response into clean market data."""
        return {
            "symbol": data.get("01. symbol"),
            "price": float(data.get("05. price", 0)),
            "change_percent": data.get("10. change percent", "0%"),
            "volume": int(data.get("06. volume", 0)),
            "last_updated": datetime.now().isoformat()
        }

    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached data if still valid."""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.cache_duration:
                return data
            del self.cache[key]
        return None

    def _add_to_cache(self, key: str, data: Dict[str, Any]):
        """Cache data with timestamp."""
        self.cache[key] = (data, datetime.now())