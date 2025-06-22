"""Service for fetching and analyzing news sentiment using NewsAPI."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date
import asyncio
import time
import hashlib
from newsapi import NewsApiClient
from sqlalchemy.orm import Session
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from ..config import get_settings
from .base_sentiment import BaseSentimentAnalyzer
from ..database.models import RawSentimentAnalysis
from ..database.database import get_db

logger = logging.getLogger(__name__)
settings = get_settings()

# Simple in-memory cache for news results
_news_cache = {}
_news_cache_expiry_seconds = 3600 * 1  # Cache news for 1 hour

class NewsSentimentService:
    """Fetches news articles, analyzes their sentiment, and stores raw analysis."""

    def __init__(self, db: AsyncSession = Depends(get_db)):
        # Use composition instead of inheritance
        self.sentiment_analyzer = BaseSentimentAnalyzer()
        self.db = db  # Store DB session
        if not settings.api_key_news:
            logger.warning("NewsAPI key not configured. NewsSentimentService will return empty results.")
            self.newsapi = None
        else:
            try:
                self.newsapi = NewsApiClient(api_key=settings.api_key_news)
            except Exception as e:
                logger.error(f"Failed to initialize NewsApiClient: {e}")
                self.newsapi = None
        self.request_delay = 1  # Basic delay between API calls if needed
        self.last_request_time = 0

    def _generate_content_hash(self, text_content: str) -> str:
        """Generates an SHA256 hash for a given text content."""
        return hashlib.sha256(text_content.encode('utf-8')).hexdigest()

    async def _store_raw_analysis(self, symbol: Optional[str], article_data: Dict, sentiment_result: Dict):
        """Stores the raw sentiment analysis of a single article into the database."""
        if "error" in sentiment_result or not sentiment_result.get("all_scores"):
            logger.warning(f"Skipping DB storage for article due to sentiment analysis error or missing scores: {article_data.get('title')}")
            return

        text_to_analyze = f"{article_data.get('title', '')}. {article_data.get('description', '') or article_data.get('content', '')}"
        content_hash = self._generate_content_hash(text_to_analyze)

        try:
            # Use async query to check for existence
            stmt = select(RawSentimentAnalysis).where(RawSentimentAnalysis.text_content_hash == content_hash)
            result = await self.db.execute(stmt)
            existing_analysis = result.scalar_one_or_none()

            if existing_analysis:
                logger.debug(f"Raw analysis for content hash {content_hash} already exists. Skipping storage.")
                return
        except Exception as e:
            logger.error(f"Error checking for existing raw analysis (hash: {content_hash}): {e}", exc_info=True)
            # Do not proceed if check fails
            return

        source_created_at_str = article_data.get('publishedAt')
        source_created_at_dt = None
        if source_created_at_str:
            try:
                source_created_at_dt = datetime.fromisoformat(source_created_at_str.replace('Z', '+00:00'))
            except ValueError:
                logger.warning(f"Could not parse article publishedAt date: {source_created_at_str}")

        raw_analysis_entry = RawSentimentAnalysis(
            symbol=symbol,
            text_content_hash=content_hash,
            text_content=text_to_analyze,
            source="NewsAPI_" + (article_data.get('source', {}).get('id') or article_data.get('source', {}).get('name', 'unknown')).replace(" ", "_"),
            sentiment_label=sentiment_result.get("primary_sentiment", "neutral"),
            sentiment_score=sentiment_result.get("primary_confidence", 0.0),
            all_scores=sentiment_result.get("all_scores"),
            analyzed_at=datetime.utcnow(),
            source_created_at=source_created_at_dt
        )
        try:
            self.db.add(raw_analysis_entry)
            # The session will be committed by the calling service (SentimentAnalysisService)
            # await self.db.commit()
            logger.debug(f"Added raw news analysis to session for: {article_data.get('title', '')[:50]}...")
        except Exception as e:
            logger.error(f"DB Error adding raw news analysis to session for '{article_data.get('title', '')[:50]}...': {e}", exc_info=True)
            # Rollback should also be handled by the top-level service.
            # await self.db.rollback()
            # Re-raise the exception so the caller knows the operation failed.
            raise

    async def get_news_sentiment(self, symbol: str, company_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get aggregated sentiment from recent news articles related to a symbol.
        Also stores individual article analyses.
        """
        if not self.newsapi:
            return self._create_empty_sentiment(symbol, "newsapi_key_missing")

        cache_key = f"news_sentiment_agg_v2:{symbol}"
        cached_data = self._get_cached_news(cache_key)
        if cached_data:
            logger.info(f"Returning cached aggregated news sentiment for {symbol}")
            return cached_data

        try:
            query = self._build_query(symbol, company_name)
            logger.info(f"Fetching news for query: '{query}'")
            await self._respect_rate_limit()
            to_date = datetime.now()
            from_date = to_date - timedelta(days=7)

            articles_response = await asyncio.to_thread(
                self.newsapi.get_everything,
                q=query, language='en', sort_by='relevancy',
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'), page_size=20
            )
            self.last_request_time = time.time()

            if articles_response['status'] != 'ok' or not articles_response['articles']:
                return self._create_empty_sentiment(symbol, "no_articles_found")

            articles = articles_response['articles']
            processed_sentiments_for_aggregation = []
            storage_tasks = []

            for article in articles:
                title = article.get('title')
                content_preview = article.get('description', '') or article.get('content', '')
                if not title and not content_preview:
                    continue
                text_to_analyze = f"{title}. {content_preview}"
                
                sentiment_result = await self.sentiment_analyzer.analyze_text(text_to_analyze)
                
                if "error" not in sentiment_result:
                    processed_sentiments_for_aggregation.append(sentiment_result)
                    storage_tasks.append(self._store_raw_analysis(symbol, article, sentiment_result))
                else:
                    logger.warning(f"Sentiment analysis failed for article: {title}")
            
            if storage_tasks:
                storage_results = await asyncio.gather(*storage_tasks, return_exceptions=True)
                for res in storage_results:
                    if isinstance(res, Exception):
                        logger.error(f"An error occurred during raw news analysis DB storage: {res}", exc_info=res)
                        # We don't commit here, but we raise to signal a failure in the unit of work.
                        # The top-level handler will perform a rollback.
                        raise HTTPException(status_code=500, detail="Failed to store raw sentiment data.")

            aggregated_result = self._aggregate_news_sentiment(symbol, articles, processed_sentiments_for_aggregation)
            self._cache_news(cache_key, aggregated_result)
            return aggregated_result

        except Exception as e:
            logger.error(f"Error in get_news_sentiment for {symbol}: {e}", exc_info=True)
            return self._create_empty_sentiment(symbol, f"error: {str(e)}")

    def _build_query(self, symbol: str, company_name: Optional[str] = None) -> str:
        """Build a search query for NewsAPI."""
        if company_name:
            query_term = f'"{company_name}"' if ' ' in company_name else company_name
            query = f'({query_term} OR {symbol}) AND (stock OR market OR finance OR business)'
        else:
            query = f'{symbol} AND (stock OR market OR finance OR business)'
        return query

    async def _respect_rate_limit(self):
        """Basic rate limiting for NewsAPI calls."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.request_delay:
            await asyncio.sleep(self.request_delay - time_since_last)

    def _aggregate_news_sentiment(
        self, symbol: str, articles: List[Dict], sentiments: List[Dict]
    ) -> Dict[str, Any]:
        """Aggregate sentiment scores from multiple articles."""
        if not sentiments:
            return self._create_empty_sentiment(symbol, "no_sentiments_analyzed")

        valid_sentiments = [s for s in sentiments if s and 'sentiment' in s]
        if not valid_sentiments:
             return self._create_empty_sentiment(symbol, "all_sentiment_analyses_failed")

        total_score = 0
        weighted_score_sum = 0
        total_weight = 0
        sentiment_counts = {"bullish": 0, "bearish": 0, "neutral": 0}

        for sentiment in valid_sentiments:
            score = 0
            if sentiment['sentiment'] == 'bullish':
                score = sentiment['confidence']
                sentiment_counts["bullish"] += 1
            elif sentiment['sentiment'] == 'bearish':
                score = -sentiment['confidence']
                sentiment_counts["bearish"] += 1
            else:
                sentiment_counts["neutral"] += 1
            
            weight = sentiment['confidence']
            weighted_score_sum += score * weight
            total_weight += weight
            total_score += score

        normalized_score = (weighted_score_sum / total_weight) if total_weight > 0.1 else (total_score / len(valid_sentiments))

        if normalized_score > 0.15:
            overall_sentiment = "bullish"
        elif normalized_score < -0.15:
            overall_sentiment = "bearish"
        else:
            overall_sentiment = "neutral"

        avg_confidence = sum(s['confidence'] for s in valid_sentiments) / len(valid_sentiments)

        top_articles_data = []
        for i, article in enumerate(articles):
             if i < len(valid_sentiments):
                 sentiment = valid_sentiments[i]
                 top_articles_data.append({
                     "title": article.get('title'),
                     "url": article.get('url'),
                     "publishedAt": article.get('publishedAt'),
                     "source": article.get('source', {}).get('name'),
                     "sentiment": sentiment['sentiment'],
                     "confidence": sentiment['confidence']
                 })
        top_articles_data.sort(key=lambda x: x['confidence'], reverse=True)

        return {
            "symbol": symbol,
            "sentiment": overall_sentiment,
            "normalized_score": round(normalized_score, 3),
            "confidence": round(avg_confidence, 3),
            "article_count": len(valid_sentiments),
            "sentiment_distribution": sentiment_counts,
            "top_articles": top_articles_data[:5],
            "last_updated": datetime.now().isoformat(),
            "source": "newsapi"
        }

    def _create_empty_sentiment(self, symbol: str, reason: str = "unknown") -> Dict[str, Any]:
        """Return an empty sentiment structure."""
        logger.warning(f"Returning empty news sentiment for {symbol}. Reason: {reason}")
        return {
            "symbol": symbol,
            "sentiment": "neutral",
            "normalized_score": 0.0,
            "confidence": 0.0,
            "article_count": 0,
            "sentiment_distribution": {"bullish": 0, "bearish": 0, "neutral": 0},
            "top_articles": [],
            "last_updated": datetime.now().isoformat(),
            "source": "newsapi",
            "status": f"No data ({reason})"
        }

    def _get_cached_news(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Get news data from in-memory cache if not expired."""
        if cache_key in _news_cache:
            data, timestamp = _news_cache[cache_key]
            if time.time() - timestamp < _news_cache_expiry_seconds:
                return data
            else:
                del _news_cache[cache_key]
        return None

    def _cache_news(self, cache_key: str, data: Dict[str, Any]):
        """Cache news data in memory with a timestamp."""
        _news_cache[cache_key] = (data, time.time())
        logger.debug(f"Cached news data for key: {cache_key}")
