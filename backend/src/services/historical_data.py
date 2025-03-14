from typing import List, Dict, Any
import aiohttp
from datetime import datetime, timedelta
from ..config import get_settings
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

class HistoricalDataService:
    def __init__(self):
        self.api_key = settings.api_key_alpha_vantage
        self.base_url = "https://www.alphavantage.co/query"

    async def get_historical_prices(
        self,
        symbol: str,
        interval: str = "daily",
        output_size: str = "compact"
    ) -> List[Dict[str, Any]]:
        """Get historical price data for charting."""
        params = {
            "function": "TIME_SERIES_DAILY",
            "symbol": symbol,
            "apikey": self.api_key,
            "outputsize": output_size
        }

        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url, params=params) as response:
                data = await response.json()
                if "Time Series (Daily)" not in data:
                    logger.error(f"No data found for {symbol}")
                    return []

                time_series = data["Time Series (Daily)"]
                return [
                    {
                        "date": date,
                        "open": float(values["1. open"]),
                        "high": float(values["2. high"]),
                        "low": float(values["3. low"]),
                        "close": float(values["4. close"]),
                        "volume": int(values["5. volume"])
                    }
                    for date, values in time_series.items()
                ]

    async def get_sentiment_history(
        self,
        symbol: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get historical sentiment data."""
        # This would typically come from your database
        # For now, return mock data
        return [
            {
                "date": (datetime.now() - timedelta(days=i)).isoformat(),
                "sentiment": "bullish" if i % 3 == 0 else "bearish" if i % 3 == 1 else "neutral",
                "confidence": 0.7 + (i % 3) * 0.1
            }
            for i in range(days)
        ]

    async def get_volatility_history(
        self,
        symbol: str,
        days: int = 30
    ) -> List[Dict[str, Any]]:
        """Get historical volatility data."""
        price_data = await self.get_historical_prices(symbol, output_size="full")
        if not price_data:
            return []

        # Calculate historical volatility for each period
        volatilities = []
        for i in range(len(price_data) - 20):  # 20-day rolling window
            window = price_data[i:i+20]
            prices = [day["close"] for day in window]
            volatility = self._calculate_volatility(prices)
            volatilities.append({
                "date": price_data[i]["date"],
                "volatility": volatility,
                "price": price_data[i]["close"]
            })

        return volatilities[-days:]  # Return only requested number of days

    def _calculate_volatility(self, prices: List[float]) -> float:
        """Calculate volatility from a list of prices."""
        if len(prices) < 2:
            return 0.0
        
        returns = [
            (prices[i] - prices[i-1]) / prices[i-1]
            for i in range(1, len(prices))
        ]
        
        mean = sum(returns) / len(returns)
        variance = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
        return (variance ** 0.5) * (252 ** 0.5)  # Annualized volatility

    async def get_market_analysis(self, symbol: str) -> Dict[str, Any]:
        """Provide comprehensive market analysis."""
        prices = await self.get_historical_prices(symbol)
        volatility = await self.get_volatility_history(symbol)
        sentiment = await self.get_sentiment_history(symbol)

        # Calculate trends
        price_trend = self._calculate_trend([p["close"] for p in prices[-5:]])
        vol_trend = self._calculate_trend([v["volatility"] for v in volatility[-5:]])
        
        # Calculate key levels
        price_data = [float(p["close"]) for p in prices]
        support = min(price_data[-20:])
        resistance = max(price_data[-20:])
        
        return {
            "trends": {
                "price": price_trend,
                "volatility": vol_trend,
                "sentiment": self._analyze_sentiment_trend(sentiment[-5:])
            },
            "key_levels": {
                "support": support,
                "resistance": resistance,
                "current_price": price_data[-1],
                "distance_to_support": price_data[-1] - support,
                "distance_to_resistance": resistance - price_data[-1]
            },
            "volatility_analysis": {
                "current": volatility[-1]["volatility"] if volatility else None,
                "trend": vol_trend,
                "is_increasing": vol_trend > 0
            },
            "timestamp": datetime.now().isoformat()
        }

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate the trend slope using linear regression."""
        if not values:
            return 0
        x = list(range(len(values)))
        n = len(values)
        sum_x = sum(x)
        sum_y = sum(values)
        sum_xy = sum(x[i] * values[i] for i in range(n))
        sum_xx = sum(x[i] * x[i] for i in range(n))
        
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
        return slope

    def _analyze_sentiment_trend(self, sentiments: List[Dict[str, Any]]) -> str:
        """Analyze sentiment trend."""
        if not sentiments:
            return "neutral"
            
        bullish_count = sum(1 for s in sentiments if s["sentiment"] == "bullish")
        bearish_count = sum(1 for s in sentiments if s["sentiment"] == "bearish")
        
        if bullish_count > len(sentiments) // 2:
            return "improving"
        elif bearish_count > len(sentiments) // 2:
            return "deteriorating"
        return "stable"
