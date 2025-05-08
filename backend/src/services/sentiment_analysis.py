"""Service for aggregating sentiment from multiple sources."""

import logging
from typing import Dict, Any, Optional
from datetime import date, datetime, timedelta
import asyncio
from sqlalchemy.orm import Session
from sqlalchemy import select, and_
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException

from .base_sentiment import BaseSentimentAnalyzer
from .social_sentiment import SocialSentimentService
from .news_sentiment import NewsSentimentService
from .volatility import VolatilityService
from ..database.models import AggregatedSentiment, SentimentType
from ..schemas.sentiment import AggregatedSentimentResponse, SentimentCreate, VolatilityContext
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

class SentimentAnalysisService(BaseSentimentAnalyzer):
    """Aggregates sentiment from news, social media, and potentially other sources."""

    def __init__(self, db: Session):
        """Initialize with DB session and sub-services."""
        super().__init__()
        self.db = db
        self.social_service = SocialSentimentService()
        self.news_service = NewsSentimentService()
        self.volatility_service = VolatilityService()
        # Define weights for aggregation
        self.source_weights = {
            "news_sentiment": 0.6,
            "reddit_sentiment": 0.4
        }

    async def get_aggregated_sentiment(
        self, symbol: str, target_date: date = date.today()
    ) -> AggregatedSentimentResponse:
        """
        Fetches sentiment from all sources, aggregates them, stores in DB,
        and returns the result including volatility context.

        Args:
            symbol: The stock/asset symbol.
            target_date: The date for which to aggregate sentiment.

        Returns:
            AggregatedSentimentResponse containing the combined sentiment and context.

        Raises:
            HTTPException: If data fetching, aggregation, or DB operations fail.
        """
        logger.info(f"Aggregating sentiment for {symbol} on {target_date}")

        # 1. Check DB first for existing aggregated data for the target date
        db_sentiment = self._get_sentiment_from_db(symbol, target_date)
        if db_sentiment:
            logger.info(f"Returning existing aggregated sentiment for {symbol} on {target_date} from DB.")
            # Add volatility context before returning
            volatility_data = await self.volatility_service.calculate_and_predict_volatility(symbol)
            return self._create_response_with_context(db_sentiment, volatility_data)

        # 2. Fetch data from sources concurrently
        news_task = self.news_service.get_news_sentiment(symbol)
        social_task = self.social_service.get_reddit_sentiment(symbol)
        volatility_task = self.volatility_service.calculate_and_predict_volatility(symbol)

        news_result, social_result, volatility_result = await asyncio.gather(
            news_task, social_task, volatility_task, return_exceptions=True
        )

        # Handle potential errors during fetching
        if isinstance(news_result, Exception):
            logger.error(f"Failed to fetch news sentiment for {symbol}: {news_result}")
            news_result = None
        if isinstance(social_result, Exception):
            logger.error(f"Failed to fetch social sentiment for {symbol}: {social_result}")
            social_result = None
        if isinstance(volatility_result, Exception):
             logger.error(f"Failed to fetch volatility for {symbol}: {volatility_result}")
             volatility_result = None

        # 3. Aggregate the results
        if not news_result and not social_result:
            logger.error(f"No sentiment data available from any source for {symbol}. Cannot aggregate.")
            raise HTTPException(status_code=404, detail=f"No sentiment data found for {symbol}")

        aggregated_data = self._perform_aggregation(symbol, target_date, news_result, social_result)

        # 4. Store the aggregated result in the database
        try:
            stored_sentiment = self._store_sentiment_in_db(aggregated_data)
            logger.info(f"Stored aggregated sentiment for {symbol} on {target_date} in DB.")
            # 5. Return response with volatility context
            return self._create_response_with_context(stored_sentiment, volatility_result)
        except SQLAlchemyError as e:
            logger.error(f"Failed to store aggregated sentiment for {symbol}: {e}")
            raise HTTPException(status_code=500, detail="Database error storing sentiment.")
        except Exception as e:
            logger.error(f"Unexpected error after aggregation for {symbol}: {e}")
            raise HTTPException(status_code=500, detail="Server error processing sentiment.")

    def _perform_aggregation(
        self, symbol: str, target_date: date,
        news_result: Optional[Dict], social_result: Optional[Dict]
    ) -> SentimentCreate:
        """Combines sentiment scores from different sources using defined weights."""
        
        weighted_score_sum = 0
        total_weight = 0
        contributing_sources = {}

        if news_result and news_result.get('article_count', 0) > 0:
            weight = self.source_weights['news_sentiment']
            score = news_result['normalized_score']
            weighted_score_sum += score * weight
            total_weight += weight
            contributing_sources['news_sentiment'] = news_result

        reddit_data_to_use = None
        if social_result and social_result.get('timeframes'):
            reddit_data_to_use = social_result['timeframes'].get('24h')
        
        if reddit_data_to_use and reddit_data_to_use.get('post_count', 0) > 0:
            weight = self.source_weights['reddit_sentiment']
            reddit_score = 0
            if reddit_data_to_use['sentiment'] == 'bullish':
                reddit_score = reddit_data_to_use['confidence']
            elif reddit_data_to_use['sentiment'] == 'bearish':
                reddit_score = -reddit_data_to_use['confidence']
            
            weighted_score_sum += reddit_score * weight
            total_weight += weight
            contributing_sources['reddit_sentiment'] = social_result

        if total_weight == 0:
            final_score = 0.0
        else:
            final_score = weighted_score_sum / total_weight

        if final_score > 0.15:
            final_sentiment_label = SentimentType.BULLISH
        elif final_score < -0.15:
            final_sentiment_label = SentimentType.BEARISH
        else:
            final_sentiment_label = SentimentType.NEUTRAL

        sentiment_to_store = SentimentCreate(
            symbol=symbol,
            date=target_date,
            sentiment=final_sentiment_label,
            score=round(final_score, 4),
            news_score=news_result['normalized_score'] if news_result else None,
            reddit_score=reddit_score if reddit_data_to_use else None,
            avg_daily_score=None,
            moving_avg_7d=None,
            benchmark=None,
            timestamp=datetime.now()
        )
        return sentiment_to_store

    def _get_sentiment_from_db(self, symbol: str, target_date: date) -> Optional[AggregatedSentiment]:
        """Retrieve aggregated sentiment from the database for a specific symbol and date."""
        try:
            stmt = select(AggregatedSentiment).where(
                and_(
                    AggregatedSentiment.symbol == symbol,
                    AggregatedSentiment.date == target_date
                )
            )
            result = self.db.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"DB error fetching sentiment for {symbol} on {target_date}: {e}")
            return None

    def _store_sentiment_in_db(self, sentiment_data: SentimentCreate) -> AggregatedSentiment:
        """Store or update aggregated sentiment in the database."""
        existing = self._get_sentiment_from_db(sentiment_data.symbol, sentiment_data.date)
        
        if existing:
            for key, value in sentiment_data.model_dump(exclude_unset=True).items():
                setattr(existing, key, value)
            existing.timestamp = datetime.now()
            db_sentiment = existing
            logger.debug(f"Updating existing sentiment record for {sentiment_data.symbol} on {sentiment_data.date}")
        else:
            db_sentiment = AggregatedSentiment(**sentiment_data.model_dump())
            self.db.add(db_sentiment)
            logger.debug(f"Creating new sentiment record for {sentiment_data.symbol} on {sentiment_data.date}")

        self.db.commit()
        self.db.refresh(db_sentiment)
        return db_sentiment

    def _create_response_with_context(
        self, 
        db_sentiment: AggregatedSentiment, 
        volatility_data: Optional[Dict]
    ) -> AggregatedSentimentResponse:
        """Combines DB sentiment data with volatility context into the response schema."""
        
        vol_context = VolatilityContext(level=0.0, is_high=False, trend="unknown")
        market_condition = "unknown"
        if volatility_data:
            vol_context = VolatilityContext(
                level=volatility_data.get('current_volatility', 0.0),
                is_high=volatility_data.get('is_high_volatility', False),
                trend=volatility_data.get('trend', 'unknown')
            )
            market_condition = volatility_data.get('market_conditions', 'unknown')

        response = AggregatedSentimentResponse(
            id=db_sentiment.id,
            symbol=db_sentiment.symbol,
            date=db_sentiment.date,
            overall_sentiment=db_sentiment.sentiment,
            normalized_score=db_sentiment.score,
            avg_daily_score=db_sentiment.avg_daily_score,
            moving_avg_7d=db_sentiment.moving_avg_7d,
            benchmark=db_sentiment.benchmark,
            market_condition=market_condition,
            volatility_context=vol_context,
            source_weights=self.source_weights,
            timestamp=db_sentiment.timestamp
        )
        return response

    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyzes a single piece of text using the base sentiment analyzer.
        Overrides base method if needed, or relies on inherited method.
        This method now primarily serves the sub-services (News, Social).
        The API endpoint /analyze-text uses this.

        Args:
            text: The text to analyze.

        Returns:
            A dictionary containing sentiment label and confidence score.
        """
        result = await super().analyze_text(text)
        result['timestamp'] = datetime.now().isoformat()
        return result