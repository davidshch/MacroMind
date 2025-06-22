"""Services for interacting with market data providers."""

import aiohttp
import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, date
import finnhub
from ..config import get_settings
import logging
import os
import numpy as np

logger = logging.getLogger(__name__)
settings = get_settings()

def generate_historical_price_data(symbol: str, start_date: date, end_date: date) -> list:
    """Generate realistic historical price data for training."""
    
    base_prices = {
        "NVDA": 150.0, "AAPL": 180.0, "TSLA": 250.0, "MSFT": 400.0,
        "GOOGL": 140.0, "AMZN": 180.0, "META": 500.0, "SPY": 450.0,
        "QQQ": 380.0, "^VIX": 20.0
    }
    base_price = base_prices.get(symbol, 100.0)
    
    trading_days = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:
            trading_days.append(current_date)
        current_date += timedelta(days=1)
    
    price_data = []
    current_price = base_price
    
    for i, day in enumerate(trading_days):
        trend_factor = 0.0001 * (i % 252)
        volatility = 0.02
        random_factor = np.random.normal(0, volatility)
        price_change = trend_factor + random_factor
        current_price *= (1 + price_change)
        
        daily_volatility = 0.01
        open_price = current_price * (1 + np.random.normal(0, daily_volatility * 0.5))
        high_price = max(open_price, current_price) * (1 + abs(np.random.normal(0, daily_volatility * 0.3)))
        low_price = min(open_price, current_price) * (1 - abs(np.random.normal(0, daily_volatility * 0.3)))
        
        price_data.append({
            "date": day.strftime("%Y-%m-%d"), "open": round(open_price, 2),
            "high": round(high_price, 2), "low": round(low_price, 2),
            "close": round(current_price, 2), "volume": int(1000000 * (1 + np.random.normal(0, 0.3)))
        })
    return price_data

class AlphaVantageService:
    """Service for interacting with Alpha Vantage API."""
    
    def __init__(self):
        self.api_key = settings.api_key_alpha_vantage
        self.base_url = "https://www.alphavantage.co/query"
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(hours=24)  # Cache for 24 hours by default

    async def get_sector_performance(self) -> Dict[str, Any]:
        """Get real-time and historical sector performance data."""
        try:
            # Check if we're in test mode
            if os.getenv('TESTING') == 'TRUE':
                return self._get_mock_sector_performance()
                
            # Check cache
            cache_key = "sector_performance"
            cached_data = self._check_cache(cache_key)
            if cached_data:
                return cached_data
                
            params = {
                "function": "SECTOR",
                "apikey": self.api_key
            }
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        self._update_cache(cache_key, data)
                        return data
                    else:
                        raise Exception(f"Alpha Vantage API error: {response.status}")
        except Exception as e:
            logger.error(f"Error fetching sector performance: {str(e)}")
            # Return mock data or raise exception based on desired behavior
            return self._get_mock_sector_performance() # Example: return mock on error

    def _get_mock_sector_performance(self) -> Dict[str, Any]:
        """Return mock sector performance data for testing or error fallback."""
        # Simplified mock data structure
        return {
            "Meta Data": {"Information": "Mock Sector Performance Data"},
            "Rank A: Real-Time Performance": {
                "Technology": "1.23%",
                "Healthcare": "0.55%",
                # ... other sectors
            }
            # Add other ranks if needed
        }

    async def get_overview(self, symbol: str) -> Dict[str, Any]:
        """Get company/ETF overview including fundamentals."""
        try:
            # Check if we're in test mode
            if os.getenv('TESTING') == 'TRUE':
                return self._get_mock_overview(symbol)
                
            # Check cache
            cache_key = f"overview_{symbol}"
            cached_data = self._check_cache(cache_key)
            if cached_data:
                return cached_data
                
            params = {
                "function": "OVERVIEW",
                "symbol": symbol,
                "apikey": self.api_key
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        # Check if we got an error response or empty data
                        if "Error Message" in data or not data:
                            logger.warning(f"Alpha Vantage API returned error for {symbol}: {data}")
                            return self._get_mock_overview(symbol)
                            
                        self._update_cache(cache_key, data)
                        return data
                    else:
                        logger.error(f"Alpha Vantage API error for {symbol}: {response.status}")
                        return self._get_mock_overview(symbol)
        except Exception as e:
            logger.error(f"Error fetching overview for {symbol}: {str(e)}")
            # Return mock data on error for robustness
            return self._get_mock_overview(symbol)
    
    async def get_historical_data(self, symbol: str, period: str = "daily") -> List[Dict[str, Any]]:
        """Get historical price data for a symbol. For the demo, this uses a reliable internal generator."""
        logger.info(f"DEMO MODE: Using generated historical data for {symbol}.")
        end_date = date.today()
        start_date = end_date - timedelta(days=365) # Generate a year of data
        # The data is generated oldest to newest. The calling functions will sort/slice as needed.
        return generate_historical_price_data(symbol, start_date, end_date)
            
    def _check_cache(self, key: str) -> Optional[Any]:
        """Check if data exists in cache and is not expired."""
        if key in self.cache and datetime.now() < self.cache_expiry.get(key, datetime.min):
            return self.cache[key]
        return None
        
    def _update_cache(self, key: str, data: Any):
        """Update cache with new data and expiry time."""
        self.cache[key] = data
        self.cache_expiry[key] = datetime.now() + self.cache_duration
        
    def _get_mock_overview(self, symbol: str) -> Dict[str, Any]:
        """Return mock company/ETF overview data for testing."""
        # Standard mock data for any symbol
        mock_data = {
            "Symbol": symbol,
            "AssetType": "ETF",
            "Name": f"{symbol} Sector ETF",
            "Description": f"This is a mock description for {symbol} ETF used for testing.",
            "Exchange": "NYSE",
            "Currency": "USD",
            "PERatio": "22.5",
            "PEGRatio": "2.1",
            "PriceToBookRatio": "3.2",
            "EPS": "2.75",
            "EPSGrowth": "0.12",
            "DividendYield": "1.8",
            "MarketCapitalization": "25000000000",
            "52WeekHigh": "180.25",
            "52WeekLow": "120.40",
            "50DayMovingAverage": "155.75",
            "200DayMovingAverage": "145.30"
        }
        
        # Add some sector-specific variations
        if symbol == "XLK" or symbol.lower() == "technology":
            mock_data.update({
                "Name": "Technology Select Sector SPDR Fund",
                "PERatio": "28.5",
                "PriceToBookRatio": "6.8",
                "EPSGrowth": "0.18"
            })
        elif symbol == "XLV" or symbol.lower() == "healthcare":
            mock_data.update({
                "Name": "Health Care Select Sector SPDR Fund",
                "PERatio": "24.2",
                "PriceToBookRatio": "4.5",
                "EPSGrowth": "0.11"
            })
        elif symbol == "XLF" or symbol.lower() == "financials":
            mock_data.update({
                "Name": "Financial Select Sector SPDR Fund",
                "PERatio": "14.8",
                "PriceToBookRatio": "1.5",
                "EPSGrowth": "0.08"
            })
            
        return mock_data
        
    def _get_mock_historical_data(self, symbol: str) -> List[Dict[str, Any]]:
        """Return mock historical price data for testing."""
        # Generate 30 days of mock data
        data = []
        base_price = 150.0 if symbol not in ["XLE", "XLU"] else 75.0
        
        for i in range(30, 0, -1):
            date = (datetime.now() - timedelta(days=i)).strftime("%Y-%m-%d")
            close_price = base_price * (1 + 0.002 * (i % 5 - 2))  # Small oscillation
            
            data.append({
                "date": date,
                "open": close_price * 0.995,
                "high": close_price * 1.01,
                "low": close_price * 0.99,
                "close": close_price,
                "volume": 1000000 + (i % 7) * 100000
            })
            
            # Update base price with a slight trend
            base_price *= (1 + 0.001 * (i % 3 - 1))
            
        return data


class FinnhubService:
    """Service for interacting with Finnhub API."""
    
    def __init__(self):
        self.api_key = settings.finnhub_api_key
        if self.api_key and not os.getenv('TESTING') == 'TRUE':
            self.client = finnhub.Client(api_key=self.api_key)
        else:
            self.client = None
        
        self.cache = {}
        self.cache_expiry = {}
        self.cache_duration = timedelta(hours=12)  # Cache for 12 hours by default

    async def get_company_metrics(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company financial metrics from Finnhub."""
        try:
            # Check if we're in test mode
            if os.getenv('TESTING') == 'TRUE':
                return self._get_mock_company_metrics(symbol)
                
            # Check cache
            cache_key = f"metrics_{symbol}"
            cached_data = self._check_cache(cache_key)
            if cached_data:
                return cached_data
                
            if not self.client:
                return self._get_mock_company_metrics(symbol)
                
            # Using asyncio.to_thread since finnhub client is synchronous
            data = await asyncio.to_thread(self.client.company_basic_financials, symbol, 'all')
            
            if not data or 'metric' not in data:
                return self._get_mock_company_metrics(symbol)
                
            self._update_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error fetching company metrics for {symbol}: {str(e)}")
            return self._get_mock_company_metrics(symbol)
            
    async def get_company_profile(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get company profile information from Finnhub."""
        try:
            # Check if we're in test mode
            if os.getenv('TESTING') == 'TRUE':
                return self._get_mock_company_profile(symbol)
                
            # Check cache
            cache_key = f"profile_{symbol}"
            cached_data = self._check_cache(cache_key)
            if cached_data:
                return cached_data
                
            if not self.client:
                return self._get_mock_company_profile(symbol)
                
            # Using asyncio.to_thread since finnhub client is synchronous
            data = await asyncio.to_thread(self.client.company_profile2, symbol=symbol)
            
            if not data:
                return self._get_mock_company_profile(symbol)
                
            self._update_cache(cache_key, data)
            return data
        except Exception as e:
            logger.error(f"Error fetching company profile for {symbol}: {str(e)}")
            return self._get_mock_company_profile(symbol)
            
    def _check_cache(self, key: str) -> Optional[Any]:
        """Check if data exists in cache and is not expired."""
        if key in self.cache and key in self.cache_expiry:
            if datetime.now() < self.cache_expiry[key]:
                return self.cache[key]
            else:
                # Remove expired data
                del self.cache[key]
                del self.cache_expiry[key]
        return None
        
    def _update_cache(self, key: str, data: Any):
        """Update cache with new data and set expiry."""
        self.cache[key] = data
        self.cache_expiry[key] = datetime.now() + self.cache_duration
        
    def _get_mock_company_metrics(self, symbol: str) -> Dict[str, Any]:
        """Return mock company metrics data for testing."""
        return {
            "metric": {
                "10DayAverageTradingVolume": 29.40765,
                "52WeekHigh": 180.10,
                "52WeekLow": 120.50,
                "52WeekHighDate": "2025-04-15",
                "52WeekLowDate": "2024-05-15",
                "peNormalizedAnnual": 25.6,
                "peBasicExclExtraTTM": 23.4,
                "peInclExtraTTM": 22.8,
                "pbAnnual": 4.2,
                "pbQuarterly": 4.5,
                "epsGrowthTTMYoy": 0.15,
                "epsGrowth5Y": 0.22,
                "epsGrowthQuarterlyYoy": 0.18,
                "returnOnEquityTTM": 0.25,
                "returnOnAssetsTTM": 0.12,
                "totalDebt/totalEquityAnnual": 0.45,
                "dividendYieldIndicatedAnnual": 0.018
            },
            "metricType": "all",
            "symbol": symbol
        }
        
    def _get_mock_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Return mock company profile data for testing."""
        return {
            "country": "US",
            "currency": "USD",
            "exchange": "NASDAQ",
            "ipo": "1980-12-12",
            "marketCapitalization": 2500000000000,
            "name": f"{symbol} Technology Corporation",
            "phone": "123-456-7890",
            "shareOutstanding": 16500000000,
            "ticker": symbol,
            "weburl": f"http://www.{symbol.lower()}.com",
            "logo": f"https://static.finnhub.io/logo/{symbol.lower()}.png",
            "finnhubIndustry": "Technology"
        }