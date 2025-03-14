import finnhub
import logging
from typing import Dict, Any, List
from datetime import datetime, timedelta
import asyncio
from ..config import get_settings
from fastapi import HTTPException

logger = logging.getLogger(__name__)
settings = get_settings()

class EnhancedMarketDataService:
    def __init__(self):
        self.finnhub_client = finnhub.Client(api_key=settings.finnhub_api_key)
        self.cache = {}
        self.cache_duration = timedelta(minutes=5)

    async def get_company_profile(self, symbol: str) -> Dict[str, Any]:
        """Get detailed company profile from Finnhub."""
        cache_key = f"profile_{symbol}"
        if cached := self._get_from_cache(cache_key):
            return cached

        try:
            # Run Finnhub API call in a thread pool
            profile = await asyncio.to_thread(
                self.finnhub_client.company_profile2,
                symbol=symbol
            )
            if not profile:
                raise HTTPException(status_code=404, detail="Company not found")
            
            self._add_to_cache(cache_key, profile)
            return profile
        except Exception as e:
            logger.error(f"Error fetching company profile: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_company_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get company financial metrics and ratios."""
        cache_key = f"metrics_{symbol}"
        if cached := self._get_from_cache(cache_key):
            return cached

        try:
            metrics = await asyncio.to_thread(
                self.finnhub_client.company_basic_financials,
                symbol,
                'all'
            )
            if not metrics:
                raise HTTPException(status_code=404, detail="Metrics not found")
            
            self._add_to_cache(cache_key, metrics)
            return metrics
        except Exception as e:
            logger.error(f"Error fetching company metrics: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_earnings_calendar(self, symbol: str) -> List[Dict[str, Any]]:
        """Get company earnings calendar."""
        try:
            # Finnhub requires dates in YYYY-MM-DD format without time
            from_date = datetime.now().strftime('%Y-%m-%d')
            to_date = (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')
            
            calendar = await asyncio.to_thread(
                self.finnhub_client.company_earnings,  # Use company_earnings instead
                symbol
            )
            
            if not calendar:
                return []
                
            # Process and format the earnings data
            return [{
                "date": earning.get("date", ""),
                "quarter": earning.get("quarter", ""),
                "year": earning.get("year", ""),
                "estimate_eps": earning.get("epsEstimate", 0),
                "actual_eps": earning.get("epsActual", 0),
                "surprise": earning.get("surprisePercent", 0)
            } for earning in calendar]
        except Exception as e:
            logger.error(f"Error fetching earnings data: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_company_news(self, symbol: str) -> List[Dict[str, Any]]:
        """Get company-specific news."""
        try:
            from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
            to_date = datetime.now().strftime('%Y-%m-%d')
            
            news = await asyncio.to_thread(
                self.finnhub_client.company_news,
                symbol,
                _from=from_date,
                to=to_date
            )
            
            return [{
                "datetime": datetime.fromtimestamp(n.get("datetime", 0)),
                "headline": n.get("headline", ""),
                "summary": n.get("summary", ""),
                "source": n.get("source", ""),
                "url": n.get("url", ""),
                "category": n.get("category", "")
            } for n in news[:10]]  # Limit to 10 most recent news items
        except Exception as e:
            logger.error(f"Error fetching company news: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_recommendation_trends(self, symbol: str) -> List[Dict[str, Any]]:
        """Get analyst recommendations trends."""
        try:
            trends = await asyncio.to_thread(
                self.finnhub_client.recommendation_trends,
                symbol
            )
            return trends
        except Exception as e:
            logger.error(f"Error fetching recommendation trends: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_peer_companies(self, symbol: str) -> List[str]:
        """Get peer companies."""
        try:
            peers = await asyncio.to_thread(
                self.finnhub_client.company_peers,
                symbol
            )
            return peers
        except Exception as e:
            logger.error(f"Error fetching peer companies: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def _get_from_cache(self, key: str) -> Any:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.cache_duration:
                return data
            del self.cache[key]
        return None

    def _add_to_cache(self, key: str, data: Any):
        self.cache[key] = (data, datetime.now())
