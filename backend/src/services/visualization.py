from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from datetime import date, timedelta, datetime
from ..database.models import AggregatedSentiment, MarketSentiment
from .market_data import MarketDataService
import logging
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class VisualizationService:
    def __init__(self, db: AsyncSession, market_data_service: MarketDataService):
        self.db = db
        self.market_data_service = market_data_service

    async def get_historical_prices(self, symbol: str, days: int) -> List[Dict[str, Any]]:
        """
        Fetches historical price data for visualization.
        """
        try:
            # Use the injected MarketDataService
            price_data = await self.market_data_service.get_historical_prices(symbol, days)

            # Sort the data chronologically (oldest to newest)
            sorted_data = sorted(
                [p for p in price_data if p.get("date") and p.get("close") is not None],
                key=lambda p: datetime.strptime(p['date'], '%Y-%m-%d')
            )

            # The frontend expects 'price'. Let's remap 'close' to 'price'.
            return [
                {"date": p.get("date"), "price": p.get("close")}
                for p in sorted_data
            ]
        except Exception as e:
            logger.exception(f"Error fetching historical prices from MarketDataService for {symbol}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error")

    async def get_sentiment_history(self, symbol: str, days: int) -> List[Dict[str, Any]]:
        """Fetches historical sentiment scores for visualization."""
        try:
            end_date = date.today()
            start_date = end_date - timedelta(days=days)

            stmt = (
                select(AggregatedSentiment)
                .where(AggregatedSentiment.symbol == symbol)
                .where(AggregatedSentiment.date.between(start_date, end_date))
                .order_by(AggregatedSentiment.date.asc())
            )
            result = await self.db.execute(stmt)
            sentiments = result.scalars().all()

            return [{"date": rec.date.isoformat(), "score": rec.score} for rec in sentiments]
        except Exception as e:
            logger.exception(f"Error fetching sentiment history for {symbol}: {e}")
            raise HTTPException(status_code=500, detail="Internal server error") 