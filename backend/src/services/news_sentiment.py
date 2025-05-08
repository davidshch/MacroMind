"""Service for fetching and analyzing news sentiment using NewsAPI."""

import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import asyncio
import time
from newsapi import NewsApiClient

from ..config import get_settings
from .base_sentiment import BaseSentimentAnalyzer

logger = logging.getLogger(__name__)
settings = get_settings()

# Simple in-memory cache for news results
_news_cache = {}
_news_cache_expiry_seconds = 3600 * 4 # Cache news for 4 hours

class NewsSentimentService(BaseSentimentAnalyzer):
    """Fetches news articles and analyzes their sentiment."""

    def __init__(self):
        super().__init__() # Initializes the sentiment analyzer (FinBERT)
        if not settings.news_api_key:
            logger.warning("NewsAPI key not configured. NewsSentimentService will return empty results.")
            self.newsapi = None
        else:
            try:
                self.newsapi = NewsApiClient(api_key=settings.news_api_key)
            except Exception as e:
                logger.error(f"Failed to initialize NewsApiClient: {e}")
                self.newsapi = None
        self.request_delay = 1 # Basic delay between API calls if needed
        self.last_request_time = 0

    async def get_news_sentiment(self, symbol: str, company_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get aggregated sentiment from recent news articles related to a symbol.

        Args:
            symbol: The stock symbol (e.g., AAPL).
            company_name: The full company name for better search results (e.g., Apple Inc).

        Returns:
            A dictionary containing aggregated sentiment, confidence, article count,
            and top articles.
        """
        if not self.newsapi:
            return self._create_empty_sentiment(symbol, "newsapi_key_missing")

        cache_key = f"news_sentiment:{symbol}"
        cached_data = self._get_cached_news(cache_key)
        if cached_data:
            logger.info(f"Returning cached news sentiment for {symbol}")
            return cached_data

        try:
            query = self._build_query(symbol, company_name)
            logger.info(f"Fetching news for query: '{query}'")

            # Respect rate limits (basic)
            await self._respect_rate_limit()

            # Fetch news articles from the last 7 days
            to_date = datetime.now()
            from_date = to_date - timedelta(days=7)

            # Use asyncio.to_thread for the synchronous NewsAPI client call
            articles_response = await asyncio.to_thread(
                self.newsapi.get_everything,
                q=query,
                language='en',
                sort_by='relevancy', # Prioritize relevant articles
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                page_size=20 # Limit to recent relevant articles
            )

            self.last_request_time = time.time()

            if articles_response['status'] != 'ok' or not articles_response['articles']:
                logger.warning(f"No news articles found for {symbol} or API error.")
                return self._create_empty_sentiment(symbol, "no_articles_found")

            articles = articles_response['articles']

            # Analyze sentiment for each article (concurrently)
            sentiment_tasks = [
                self.analyze_text(f"{article.get('title', '')}. {article.get('description', '') or article.get('content', '')}")
                for article in articles if article.get('title') # Ensure there is content
            ]
            sentiments = await asyncio.gather(*sentiment_tasks)

            # Aggregate results
            aggregated_result = self._aggregate_news_sentiment(symbol, articles, sentiments)

            # Cache the result
            self._cache_news(cache_key, aggregated_result)

            return aggregated_result

        except Exception as e:
            logger.error(f"Error fetching or analyzing news sentiment for {symbol}: {e}")
            # Return empty sentiment on error
            return self._create_empty_sentiment(symbol, f"error: {e}")

    def _build_query(self, symbol: str, company_name: Optional[str] = None) -> str:
        """Build a search query for NewsAPI."""
        # Prefer company name if available, otherwise use symbol
        # Add terms to focus on financial/market context
        if company_name:
            # Use exact phrase matching for company name if it contains spaces
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
            
            # Simple aggregation: average score weighted by confidence
            weight = sentiment['confidence'] # Use confidence as weight
            weighted_score_sum += score * weight
            total_weight += weight
            total_score += score # For simple average calculation

        # Calculate normalized score (-1 to 1)
        # Use weighted average if weights are significant, otherwise simple average
        normalized_score = (weighted_score_sum / total_weight) if total_weight > 0.1 else (total_score / len(valid_sentiments))

        # Determine overall sentiment label
        if normalized_score > 0.15:
            overall_sentiment = "bullish"
        elif normalized_score < -0.15:
            overall_sentiment = "bearish"
        else:
            overall_sentiment = "neutral"

        # Calculate overall confidence (average confidence of individual analyses)
        avg_confidence = sum(s['confidence'] for s in valid_sentiments) / len(valid_sentiments)

        # Prepare top articles list
        top_articles_data = []
        for i, article in enumerate(articles):
             if i < len(valid_sentiments): # Ensure index is valid
                 sentiment = valid_sentiments[i]
                 top_articles_data.append({
                     "title": article.get('title'),
                     "url": article.get('url'),
                     "publishedAt": article.get('publishedAt'),
                     "source": article.get('source', {}).get('name'),
                     "sentiment": sentiment['sentiment'],
                     "confidence": sentiment['confidence']
                 })
        # Sort top articles by confidence (descending)
        top_articles_data.sort(key=lambda x: x['confidence'], reverse=True)

        return {
            "symbol": symbol,
            "sentiment": overall_sentiment,
            "normalized_score": round(normalized_score, 3),
            "confidence": round(avg_confidence, 3),
            "article_count": len(valid_sentiments),
            "sentiment_distribution": sentiment_counts,
            "top_articles": top_articles_data[:5], # Return top 5 articles
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

    # --- Caching --- 
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
