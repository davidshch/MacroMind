"""Service for aggregating sentiment from multiple sources."""

from __future__ import annotations
from typing import TYPE_CHECKING, Dict, Any, Optional, List
from datetime import date, datetime, timedelta
import logging
import asyncio
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, desc, func, exc as sqlalchemy_exc
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import numpy as np
from fastapi import HTTPException, Depends

from ..database.models import MarketSentiment, SentimentType, RawSentimentAnalysis, AggregatedSentiment
from ..schemas.sentiment import SentimentCreate, VolatilityContext, AggregatedSentimentResponse
from ..schemas.volatility import VolatilityRegime
from ..config import get_settings
from .base_sentiment import BaseSentimentAnalyzer
from .social_sentiment import SocialSentimentService
from .news_sentiment import NewsSentimentService
from .market_data import MarketDataService
from .ml.model_factory import MLModelFactory

if TYPE_CHECKING:
    from .volatility import VolatilityService  # Import inside TYPE_CHECKING

logger = logging.getLogger(__name__)
settings = get_settings()

class SentimentAnalysisService:
    """Aggregates sentiment from news, social media, and potentially other sources."""

    def __init__(self, 
                 db: AsyncSession,
                 market_data_service: MarketDataService, 
                 ml_model_factory: MLModelFactory,
                 volatility_service: VolatilityService):
        """Initialize with DB session and sub-services."""
        self.db = db
        self.social_service = SocialSentimentService(db=db)
        self.news_service = NewsSentimentService(db=db)
        self.volatility_service = volatility_service
        self.market_data_service = market_data_service
        self.ml_model_factory = ml_model_factory
        self.base_analyzer = BaseSentimentAnalyzer()  # Use composition instead of inheritance
        self.source_weights = {
            "news_sentiment": 0.6,
            "reddit_sentiment": 0.4
        }

    async def get_dynamic_benchmark(self, symbol: str = "SPY", lookback_days: int = 5, volatility_threshold: float = 0.02) -> str:
        """
        Determines the benchmark dynamically based on SPY's recent volatility.

        Args:
            symbol: The symbol to check volatility for (defaults to SPY).
            lookback_days: Number of historical days to consider for volatility calculation.
            volatility_threshold: The threshold to switch benchmarks.

        Returns:
            "RSP" if SPY's N-day volatility > threshold, otherwise "SPY".
        """
        try:
            historical_data = await self.market_data_service.get_historical_prices(symbol, lookback_days + 1)

            if not historical_data or len(historical_data) < lookback_days + 1:
                logger.warning(
                    f"Insufficient historical data for {symbol} ({len(historical_data)} points) "
                    f"to calculate {lookback_days}-day volatility. Defaulting to SPY benchmark."
                )
                return "SPY"

            closes = [item['close'] for item in reversed(historical_data)]
            closes_series = pd.Series(closes)
            log_returns = np.log(closes_series / closes_series.shift(1)).dropna()

            if len(log_returns) < lookback_days:
                logger.warning(
                    f"Not enough log returns ({len(log_returns)}) to calculate {lookback_days}-day volatility for {symbol}. "
                    f"Defaulting to SPY benchmark."
                )
                return "SPY"

            current_volatility = log_returns.std()
            logger.info(f"{lookback_days}-day log return volatility for {symbol}: {current_volatility:.4f}")

            if current_volatility > volatility_threshold:
                logger.info(f"Volatility ({current_volatility:.4f}) > threshold ({volatility_threshold}). Using RSP benchmark.")
                return "RSP"
            else:
                logger.info(f"Volatility ({current_volatility:.4f}) <= threshold ({volatility_threshold}). Using SPY benchmark.")
                return "SPY"

        except Exception as e:
            logger.error(f"Error calculating dynamic benchmark for {symbol}: {e}. Defaulting to SPY.")
            return "SPY"

    async def get_aggregated_sentiment(
        self, symbol: str, target_date: date = date.today()
    ) -> AggregatedSentimentResponse:
        """
        Fetches sentiment from all sources, aggregates them, stores in DB,
        and returns the result including volatility context.
        """
        logger.info(f"Aggregating sentiment for {symbol} on {target_date}")

        db_sentiment_record = await self._get_sentiment_from_db(symbol, target_date)
        volatility_data = await self.volatility_service.calculate_and_predict_volatility(symbol)

        # Always fetch fresh sentiment details for the response, even if we have cached data
        news_task = self.news_service.get_news_sentiment(symbol)
        social_task = self.social_service.get_reddit_sentiment(symbol)

        news_result, social_result = await asyncio.gather(
            news_task, social_task, return_exceptions=True
        )

        if isinstance(news_result, Exception):
            logger.error(f"Failed to fetch news sentiment for {symbol}: {news_result}")
            news_result = None
        if isinstance(social_result, Exception):
            logger.error(f"Failed to fetch social sentiment for {symbol}: {social_result}")
            social_result = None

        if db_sentiment_record and db_sentiment_record.moving_avg_7d is not None:
            # Check if we have valid Reddit sentiment data
            has_valid_reddit_data = (
                social_result and 
                social_result.get('post_count', 0) > 0
            )
            
            # If we don't have valid Reddit data, force a fresh fetch
            if not has_valid_reddit_data:
                logger.info(f"No valid Reddit data found for {symbol}, forcing fresh fetch")
                try:
                    # Clear the Reddit cache for this symbol
                    self.social_service.clear_cache_for_symbol(symbol)
                    
                    # Try to get fresh Reddit data
                    fresh_social_result = await self.social_service.get_reddit_sentiment(symbol)
                    if fresh_social_result and fresh_social_result.get('post_count', 0) > 0:
                        logger.info(f"Successfully fetched fresh Reddit data for {symbol}")
                        social_result = fresh_social_result
                    else:
                        logger.warning(f"Fresh Reddit fetch still returned no data for {symbol}")
                except Exception as e:
                    logger.error(f"Error during fresh Reddit fetch for {symbol}: {e}")
            
            logger.info(f"Returning existing aggregated sentiment for {symbol} on {target_date} from DB with fresh details.")
            return self._create_response_with_context(db_sentiment_record, volatility_data, news_result, social_result)

        logger.info(f"No complete sentiment data in DB for {symbol} on {target_date}. Fetching and processing.")

        selected_benchmark: str
        if volatility_data is None or isinstance(volatility_data, Exception):
            logger.warning(f"VolatilityService failed or returned no data for {symbol}. Using dynamic benchmark heuristic.")
            selected_benchmark = await self.get_dynamic_benchmark()
        else:
            selected_benchmark = self._select_benchmark(symbol, volatility_data)

        if not news_result and not social_result and not db_sentiment_record:
            logger.error(f"No sentiment data available from any source for {symbol} and no existing DB record. Cannot aggregate.")
            raise HTTPException(status_code=404, detail=f"No sentiment data found for {symbol}")

        aggregated_data_create_schema = self._perform_aggregation(
            symbol, target_date, news_result, social_result, selected_benchmark
        )

        try:
            stored_sentiment = await self._store_sentiment_in_db(aggregated_data_create_schema, symbol, target_date)
            logger.info(f"Stored/Updated aggregated sentiment for {symbol} on {target_date} in DB.")
            return self._create_response_with_context(stored_sentiment, volatility_data, news_result, social_result)
        except SQLAlchemyError as e:
            logger.error(f"Failed to store aggregated sentiment for {symbol}: {e}")
            try:
                await self.db.rollback()
            except Exception as rollback_error:
                logger.error(f"Error during rollback: {rollback_error}")
            raise HTTPException(status_code=500, detail="Database error storing sentiment.")
        except Exception as e:
            logger.error(f"Unexpected error after aggregation for {symbol}: {e}")
            try:
                await self.db.rollback()
            except Exception as rollback_error:
                logger.error(f"Error during rollback: {rollback_error}")
            raise HTTPException(status_code=500, detail="Server error processing sentiment.")

    def _select_benchmark(self, symbol: str, volatility_data: Optional[Dict]) -> str:
        """Selects a market benchmark based on volatility conditions from VolatilityService."""
        if not volatility_data:
            logger.warning(f"_select_benchmark called without volatility_data for {symbol}. Defaulting to SPY. This should ideally be handled by the caller.")
            return "SPY"

        is_high_vol = volatility_data.get('is_high_volatility', False)
        regime_str = volatility_data.get('volatility_regime', VolatilityRegime.NORMAL.value)
        
        try:
            regime = VolatilityRegime(regime_str)
        except ValueError:
            logger.warning(f"Invalid volatility regime string '{regime_str}', defaulting to NORMAL.")
            regime = VolatilityRegime.NORMAL

        if is_high_vol or regime in [VolatilityRegime.CRISIS, VolatilityRegime.HIGH_STABLE, VolatilityRegime.HIGH_INCREASING]:
            return "SPY"
        elif regime in [VolatilityRegime.LOW_STABLE, VolatilityRegime.LOW_COMPRESSION]:
            return "RSP"
        return "SPY"

    def _perform_aggregation(
        self, symbol: str, target_date: date,
        news_result: Optional[Dict], social_result: Optional[Dict],
        selected_benchmark: str
    ) -> SentimentCreate:
        """Combines sentiment scores, normalizes them, and uses the pre-selected benchmark."""
        
        weighted_score_sum = 0.0
        total_weight = 0.0
        
        news_score_normalized: Optional[float] = None
        if news_result and news_result.get('article_count', 0) > 0:
            raw_news_score = news_result.get('normalized_score')
            if isinstance(raw_news_score, (int, float)):
                news_score_normalized = max(-1.0, min(1.0, raw_news_score))
                weighted_score_sum += news_score_normalized * self.source_weights['news_sentiment']
                total_weight += self.source_weights['news_sentiment']
            else:
                logger.warning(f"News score for {symbol} is not a valid number: {raw_news_score}")

        reddit_score_normalized: Optional[float] = None
        if social_result and social_result.get('post_count', 0) > 0:
            # Use the normalized_score directly from Reddit service
            raw_reddit_score = social_result.get('normalized_score')
            if isinstance(raw_reddit_score, (int, float)):
                reddit_score_normalized = max(-1.0, min(1.0, raw_reddit_score))
                weighted_score_sum += reddit_score_normalized * self.source_weights['reddit_sentiment']
                total_weight += self.source_weights['reddit_sentiment']
            else:
                logger.warning(f"Reddit score for {symbol} is not a valid number: {raw_reddit_score}")

        if total_weight == 0:
            final_score = 0.0
        else:
            final_score = weighted_score_sum / total_weight
        
        final_score = max(-1.0, min(1.0, final_score))

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
            news_score=news_score_normalized,
            reddit_score=reddit_score_normalized,
            avg_daily_score=round(final_score, 4),
            moving_avg_7d=None,
            benchmark=selected_benchmark,
            timestamp=datetime.now()
        )
        return sentiment_to_store

    async def _get_sentiment_from_db(self, symbol: str, target_date: date) -> Optional[AggregatedSentiment]:
        """Retrieve aggregated sentiment from the database for a specific symbol and date."""
        try:
            stmt = select(AggregatedSentiment).where(
                AggregatedSentiment.symbol == symbol,
                AggregatedSentiment.date == target_date
            )
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except SQLAlchemyError as e:
            logger.error(f"DB error fetching sentiment for {symbol} on {target_date}: {e}")
            return None

    async def _store_sentiment_in_db(self, sentiment_data: SentimentCreate, symbol: str, target_date: date) -> AggregatedSentiment:
        """Store or update aggregated sentiment in the database and then update its MVA."""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                # Check if we need to rollback any pending transaction
                if self.db.is_active:
                    try:
                        await self.db.rollback()
                    except Exception as rollback_error:
                        logger.warning(f"Error during rollback attempt {attempt + 1}: {rollback_error}")
                
                db_sentiment_record = await self._get_sentiment_from_db(symbol, target_date)

                if db_sentiment_record:
                    logger.debug(f"Updating existing sentiment record for {symbol} on {target_date}")
                    db_sentiment_record.sentiment = sentiment_data.sentiment
                    db_sentiment_record.score = sentiment_data.score
                    db_sentiment_record.news_score = sentiment_data.news_score
                    db_sentiment_record.reddit_score = sentiment_data.reddit_score
                    db_sentiment_record.benchmark = sentiment_data.benchmark
                    db_sentiment_record.timestamp = datetime.now()
                else:
                    logger.debug(f"Creating new sentiment record for {symbol} on {target_date}")
                    db_sentiment_record = AggregatedSentiment(
                        symbol=symbol,
                        date=target_date,
                        sentiment=sentiment_data.sentiment,
                        score=sentiment_data.score,
                        avg_daily_score=sentiment_data.avg_daily_score,
                        moving_avg_7d=sentiment_data.moving_avg_7d,
                        news_score=sentiment_data.news_score,
                        reddit_score=sentiment_data.reddit_score,
                        benchmark=sentiment_data.benchmark,
                        timestamp=datetime.now()
                    )
                    self.db.add(db_sentiment_record)

                await self.db.commit()
                await self.db.refresh(db_sentiment_record)
                
                # Update moving average after successful commit
                await self._update_moving_average_for_record(db_sentiment_record)
                
                return db_sentiment_record
                
            except SQLAlchemyError as e:
                logger.error(f"SQLAlchemyError during commit for daily sentiment {symbol} on {target_date} (attempt {attempt + 1}): {e}")
                try:
                    await self.db.rollback()
                except Exception as rollback_error:
                    logger.error(f"Error during rollback: {rollback_error}")
                
                if attempt == max_retries - 1:
                    raise
                else:
                    # Wait before retrying
                    await asyncio.sleep(1)
                    
            except Exception as e:
                logger.error(f"Unexpected error storing sentiment for {symbol} on {target_date} (attempt {attempt + 1}): {e}")
                try:
                    await self.db.rollback()
                except Exception as rollback_error:
                    logger.error(f"Error during rollback: {rollback_error}")
                
                if attempt == max_retries - 1:
                    raise
                else:
                    # Wait before retrying
                    await asyncio.sleep(1)

    async def _update_moving_average_for_record(self, current_sentiment_record: AggregatedSentiment, mva_window_days: int = 7):
        """
        Calculates and updates the N-day moving average for the sentiment score
        directly in the database for the given record's symbol and date.

        Args:
            current_sentiment_record: The AggregatedSentiment record for which to update the MVA.
            mva_window_days: The window for the moving average (e.g., 7 for 7-day MVA).
        """
        symbol = current_sentiment_record.symbol
        current_date = current_sentiment_record.date
        logger.info(f"Updating {mva_window_days}-day MVA for {symbol} up to {current_date}")

        try:
            start_date = current_date - timedelta(days=mva_window_days - 1)
            
            stmt = select(AggregatedSentiment.date, AggregatedSentiment.avg_daily_score)\
                .where(and_(
                    AggregatedSentiment.symbol == symbol,
                    AggregatedSentiment.date >= start_date,
                    AggregatedSentiment.date <= current_date
                ))\
                .order_by(AggregatedSentiment.date.asc())
            
            results = await self.db.execute(stmt)
            recent_scores_data = results.fetchall()

            if not recent_scores_data:
                logger.warning(f"No recent scores found for {symbol} to calculate MVA up to {current_date}. Setting MVA to current day's score or None.")
                current_sentiment_record.moving_avg_7d = current_sentiment_record.avg_daily_score
            else:
                df = pd.DataFrame(recent_scores_data, columns=['date', 'avg_daily_score'])
                mva = df['avg_daily_score'].rolling(window=mva_window_days, min_periods=1).mean().iloc[-1]
                current_sentiment_record.moving_avg_7d = round(mva, 4)

        except SQLAlchemyError as e:
            logger.error(f"SQLAlchemyError updating MVA for {symbol} on {current_date}: {e}")
            # Do not rollback here, let the calling function handle it.
        except Exception as e:
            logger.error(f"Unexpected error updating MVA for {symbol} on {current_date}: {e}")
            # Do not rollback here, let the calling function handle it.

    def _create_response_with_context(
        self, 
        db_sentiment: AggregatedSentiment, 
        volatility_data: Optional[Dict],
        news_result: Optional[Dict] = None,
        social_result: Optional[Dict] = None
    ) -> AggregatedSentimentResponse:
        """Combines DB sentiment data with volatility context into the response schema."""
        
        vol_context = VolatilityContext(level=0.0, is_high=False, trend="unknown")
        market_condition_from_vol = "unknown"
        if volatility_data:
            vol_context = VolatilityContext(
                level=volatility_data.get('current_volatility', 0.0),
                is_high=volatility_data.get('is_high_volatility', False),
                trend=volatility_data.get('trend', 'unknown')
            )
            market_condition_from_vol = volatility_data.get('market_conditions', 'unknown')

        # Prepare news sentiment details
        news_sentiment_details = None
        if news_result and news_result.get('article_count', 0) > 0:
            news_sentiment_details = {
                "article_count": news_result.get('article_count', 0),
                "sentiment_distribution": news_result.get('sentiment_distribution', {}),
                "top_articles": news_result.get('top_articles', [])[:3],  # Top 3 articles
                "average_confidence": news_result.get('confidence', 0.0),
                "last_updated": news_result.get('last_updated', datetime.now().isoformat()),
                "source": "newsapi",
                "status": "success"
            }

        # Prepare Reddit sentiment details
        reddit_sentiment_details = None
        computed_reddit_score = None
        subreddit_breakdown = {}
        if social_result:
            subreddit_breakdown = social_result.get('subreddit_breakdown', {})
            computed_reddit_score = social_result.get('normalized_score')
            # Always include Reddit sentiment details, even if no posts
            reddit_sentiment_details = {
                "post_count": social_result.get('post_count', 0),
                "total_engagement": social_result.get('total_engagement', 0),
                "sentiment_distribution": social_result.get('sentiment_distribution', {}),
                "confidence": social_result.get('confidence', 0.0),
                "subreddit_breakdown": subreddit_breakdown,
                "last_updated": social_result.get('last_updated', datetime.now().isoformat()),
                "source": "reddit",
                "status": "success" if social_result.get('post_count', 0) > 0 else "no_data"
            }

        # Fallback for reddit_sentiment_score if DB value is None
        reddit_sentiment_score = db_sentiment.reddit_score
        if reddit_sentiment_score is None and computed_reddit_score is not None:
            reddit_sentiment_score = computed_reddit_score

        response = AggregatedSentimentResponse(
            id=db_sentiment.id,
            symbol=db_sentiment.symbol,
            date=db_sentiment.date,
            overall_sentiment=db_sentiment.sentiment,
            normalized_score=db_sentiment.score,
            avg_daily_score=db_sentiment.avg_daily_score,
            moving_avg_7d=db_sentiment.moving_avg_7d,
            benchmark=db_sentiment.benchmark,
            news_sentiment_details=news_sentiment_details,
            reddit_sentiment_details=reddit_sentiment_details,
            news_sentiment_score=db_sentiment.news_score,
            reddit_sentiment_score=reddit_sentiment_score,
            market_condition=market_condition_from_vol, 
            volatility_context=vol_context,
            source_weights=self.source_weights,
            timestamp=db_sentiment.timestamp
        )
        
        # Add additional context to the response
        response_dict = response.model_dump()
        
        # Calculate data quality score
        data_quality_score = 0.0
        if news_sentiment_details and news_sentiment_details.get('article_count', 0) > 0:
            data_quality_score += 0.5
        if reddit_sentiment_details and reddit_sentiment_details.get('post_count', 0) > 0:
            data_quality_score += 0.5
        
        # Calculate sentiment strength (how strong the sentiment is, regardless of direction)
        sentiment_strength = abs(db_sentiment.score)
        
        # Add summary statistics
        response_dict.update({
            "data_quality_score": round(data_quality_score, 2),
            "sentiment_strength": round(sentiment_strength, 3),
            "sentiment_confidence": round(
                (news_sentiment_details.get('average_confidence', 0.0) if news_sentiment_details else 0.0) * 0.6 +
                (reddit_sentiment_details.get('confidence', 0.0) if reddit_sentiment_details else 0.0) * 0.4, 3
            ),
            "data_sources_available": {
                "news": news_sentiment_details is not None and news_sentiment_details.get('article_count', 0) > 0,
                "reddit": reddit_sentiment_details is not None and reddit_sentiment_details.get('post_count', 0) > 0
            }
        })
        
        return AggregatedSentimentResponse(**response_dict)

    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analyzes a single piece of text using the base sentiment analyzer.
        """
        result = await self.base_analyzer.analyze_text(text)
        result['timestamp'] = datetime.now().isoformat()
        return result

    async def _fetch_raw_sentiment_data_for_insights(
        self, 
        symbol: str, 
        lookback_days: int
    ) -> List[Dict[str, Any]]:
        """
        Fetches raw sentiment analysis data (text, score, source, etc.)
        for a given symbol over a lookback period from the RawSentimentAnalysis table.
        """
        logger.info(f"Fetching raw sentiment data for {symbol} over last {lookback_days} days from DB.")
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=lookback_days)

        raw_insights_data = []
        try:
            stmt = (
                select(
                    RawSentimentAnalysis.text_content,
                    RawSentimentAnalysis.sentiment_label,
                    RawSentimentAnalysis.sentiment_score,
                    RawSentimentAnalysis.source,
                    RawSentimentAnalysis.source_created_at
                )
                .where(
                    and_(
                        RawSentimentAnalysis.symbol == symbol,
                        RawSentimentAnalysis.source_created_at >= start_date,
                        RawSentimentAnalysis.source_created_at <= end_date
                    )
                )
                .order_by(RawSentimentAnalysis.source_created_at.desc())
                .limit(50)  # Limit to a reasonable number of items for LLM context
            )
            
            result = await self.db.execute(stmt)
            fetched_rows = result.fetchall()

            for row in fetched_rows:
                raw_insights_data.append({
                    "text": row.text_content,
                    "sentiment": row.sentiment_label, # This should be "bullish", "bearish", "neutral"
                    "score": row.sentiment_score,
                    "source": row.source,
                    "published_at": row.source_created_at.isoformat() if row.source_created_at else None
                })
            
            logger.info(f"Fetched {len(raw_insights_data)} raw sentiment entries for {symbol} for insights.")

        except sqlalchemy_exc.SQLAlchemyError as e:
            logger.error(f"Database error fetching raw sentiment insights for {symbol}: {e}", exc_info=True)
        except Exception as e:
            logger.error(f"Unexpected error fetching raw sentiment insights for {symbol}: {e}", exc_info=True)
            
        return raw_insights_data

    async def get_distilled_sentiment_insights(
        self, 
        symbol: str, 
        lookback_days: int = 7, 
        num_themes: int = 3
    ) -> Dict[str, Any]:
        """
        Provides distilled sentiment insights (key themes and summary) for a symbol
        by analyzing recent raw sentiment data using an LLM via MLModelFactory.

        Args:
            symbol: The stock/asset symbol.
            lookback_days: How many days of raw sentiment data to consider.
            num_themes: The number of key themes to extract.

        Returns:
            A dictionary containing 'themes' and 'summary', or an error message.
        """
        logger.info(f"Getting distilled sentiment insights for {symbol} (lookback: {lookback_days} days, themes: {num_themes}).")
        
        raw_texts_with_sentiment = await self._fetch_raw_sentiment_data_for_insights(symbol, lookback_days)

        if not raw_texts_with_sentiment:
            logger.warning(f"No raw sentiment data found for {symbol} in the last {lookback_days} days to distill insights.")
            return {"themes": [], "summary": "No recent sentiment data available to analyze."}

        insights = await self.ml_model_factory.get_sentiment_insights(
            texts_with_sentiment=raw_texts_with_sentiment, 
            asset_symbol=symbol, 
            num_themes=num_themes
        )
        
        return insights

    async def detect_sentiment_spikes(
        self,
        symbol: str,
        lookback_days: int = 90,
        min_data_points: int = 30,
        prophet_interval_width: float = 0.95,
        prophet_changepoint_scale: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detects sentiment spikes for a given symbol using Prophet anomaly detection.

        Args:
            symbol: The stock/asset symbol.
            lookback_days: How many days of historical aggregated sentiment data to fetch.
            min_data_points: Minimum number of data points required to run detection.
            prophet_interval_width: Prophet model's interval width for anomaly detection.
            prophet_changepoint_scale: Prophet model's changepoint prior scale.

        Returns:
            A dictionary conforming to SentimentSpikeResponse or an error dict.
        """
        logger.info(f"Detecting sentiment spikes for {symbol} over last {lookback_days} days.")
        end_date = date.today()
        start_date = end_date - timedelta(days=lookback_days -1)

        try:
            stmt = select(AggregatedSentiment.date, AggregatedSentiment.score)\
                .where(and_(
                    AggregatedSentiment.symbol == symbol,
                    AggregatedSentiment.date >= start_date,
                    AggregatedSentiment.date <= end_date
                ))\
                .order_by(AggregatedSentiment.date.asc())
            
            results = await self.db.execute(stmt)
            historical_sentiment_data = results.fetchall()

            if not historical_sentiment_data or len(historical_sentiment_data) < min_data_points:
                msg = f"Not enough historical sentiment data ({len(historical_sentiment_data)} points, need {min_data_points}) for {symbol} to detect spikes."
                logger.warning(msg)
                return {"symbol": symbol, "time_period_days": lookback_days, "spikes_detected": [], "message": msg}

            df = pd.DataFrame(historical_sentiment_data, columns=['ds', 'y'])
            df['ds'] = pd.to_datetime(df['ds'])

            # Use MLModelFactory for anomaly detection
            anomaly_df = await self.ml_model_factory.detect_anomalies_with_prophet(
                data_df=df,
                interval_width=prophet_interval_width,
                changepoint_prior_scale=prophet_changepoint_scale
            )

            if anomaly_df.empty or 'is_anomaly' not in anomaly_df.columns:
                msg = f"Anomaly detection failed or returned unexpected format for {symbol}."
                logger.error(msg)
                return {"symbol": symbol, "time_period_days": lookback_days, "spikes_detected": [], "message": msg}

            spikes = []
            for _, row in anomaly_df.iterrows():
                if row['is_anomaly']:
                    spikes.append({
                        "date": row['ds'].date(),
                        "actual_score": row['y'],
                        "expected_score_upper": row['yhat_upper'],
                        "expected_score_lower": row['yhat_lower'],
                        "is_spike": True
                    })
            
            logger.info(f"Found {len(spikes)} sentiment spikes for {symbol} in the last {lookback_days} days.")
            return {
                "symbol": symbol, 
                "time_period_days": lookback_days, 
                "spikes_detected": spikes,
                "message": f"Detected {len(spikes)} spikes out of {len(anomaly_df)} data points."
            }

        except SQLAlchemyError as e:
            logger.error(f"Database error fetching sentiment for spike detection for {symbol}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Database error during spike detection.")
        except Exception as e:
            logger.error(f"Unexpected error during sentiment spike detection for {symbol}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Server error during spike detection.")