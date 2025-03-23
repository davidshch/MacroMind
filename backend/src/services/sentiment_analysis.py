import logging
from transformers import pipeline
from typing import Dict, Any, List
import aiohttp
from ..config import get_settings
from datetime import datetime, timedelta
from fastapi import HTTPException
from .base_sentiment import BaseSentimentAnalyzer
from .social_sentiment import SocialSentimentService

logger = logging.getLogger(__name__)
settings = get_settings()

"""
Sentiment Analysis Service

Implements financial sentiment analysis using FinBERT. Processes news articles,
social media content, and market data to generate sentiment predictions.

Uses:
- FinBERT for text analysis
- News API for article fetching
- Caching for API optimization
"""

class SentimentAnalysisService(BaseSentimentAnalyzer):
    """Service for analyzing market sentiment from multiple sources."""
    
    def __init__(self):
        super().__init__()
        self.cache = {}
        self.cache_duration = timedelta(hours=1)
        self.social_service = SocialSentimentService()

    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze financial text sentiment.

        Returns:
            dict: Sentiment classification and confidence score
            
        Raises:
            HTTPException: If analysis fails
        """
        try:
            result = self.sentiment_analyzer(text)[0]
            mapped_sentiment = self.sentiment_mapping[result["label"]]
            return {
                "sentiment": mapped_sentiment,
                "confidence": float(result["score"]),
                "timestamp": datetime.now().isoformat(),
                "original_text": text[:100] + "..." if len(text) > 100 else text
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            raise HTTPException(status_code=500, detail="Failed to analyze sentiment")

    async def get_market_sentiment(self, symbol: str) -> Dict[str, Any]:
        try:
            cache_key = f"sentiment_{symbol}"
            cached_data = self._get_from_cache(cache_key)
            if cached_data:
                return cached_data

            news_articles = await self._fetch_news(symbol)
            if not news_articles:
                logger.warning(f"No news articles found for {symbol}")
                return {
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "error": "No news articles found",
                    "timestamp": datetime.now().isoformat()
                }

            sentiments = []
            for article in news_articles:
                if article.get("title") and article.get("description"):
                    text = f"{article['title']} {article['description']}"
                    sentiment = await self.analyze_text(text)
                    sentiments.append({
                        **sentiment,
                        "source": article.get("source", {}).get("name"),
                        "published_at": article.get("publishedAt")
                    })

            result = self._aggregate_sentiments(sentiments)
            self._add_to_cache(cache_key, result)
            return result

        except Exception as e:
            logger.error(f"Error in get_market_sentiment: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    async def get_combined_sentiment(self, symbol: str) -> Dict[str, Any]:
        """Get combined sentiment from all sources."""
        try:
            # Get sentiment from different sources
            market_sentiment = await self.get_market_sentiment(symbol)
            social_sentiment = await self.social_service.get_reddit_sentiment(symbol)  # Changed from get_combined_social_sentiment
            
            # Updated weights for combined sources
            sources = [
                (market_sentiment, 0.6),    # News/Market weight higher
                (social_sentiment, 0.4)     # Reddit weight
            ]
            
            combined_score = 0
            for sentiment, weight in sources:
                if sentiment["sentiment"] == "bullish":
                    combined_score += weight * sentiment["confidence"]
                elif sentiment["sentiment"] == "bearish":
                    combined_score -= weight * sentiment["confidence"]
            
            # Determine overall sentiment
            overall_sentiment = "neutral"
            if combined_score > 0.2:
                overall_sentiment = "bullish"
            elif combined_score < -0.2:
                overall_sentiment = "bearish"
            
            return {
                "symbol": symbol,
                "overall_sentiment": overall_sentiment,
                "confidence_score": abs(combined_score),
                "sources": {
                    "market": market_sentiment,
                    "social": social_sentiment
                },
                "momentum_indicators": self._calculate_momentum_indicators(
                    market_sentiment, social_sentiment
                ),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error in combined sentiment analysis: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))

    def _calculate_momentum_indicators(
        self,
        market_sentiment: Dict[str, Any],
        social_sentiment: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate momentum indicators from different sentiment sources."""
        return {
            "trend_strength": self._calculate_trend_strength(
                market_sentiment, social_sentiment
            ),
            "social_momentum": social_sentiment.get("total_engagement", 0),
            "sentiment_alignment": market_sentiment["sentiment"] == social_sentiment["sentiment"]
        }

    def _calculate_trend_strength(
        self,
        market_sentiment: Dict[str, Any],
        social_sentiment: Dict[str, Any]
    ) -> str:
        """Calculate trend strength based on sentiment alignment."""
        if market_sentiment["sentiment"] == social_sentiment["sentiment"]:
            confidence = (market_sentiment["confidence"] + social_sentiment["confidence"]) / 2
            if confidence > 0.8:
                return "strong"
            elif confidence > 0.5:
                return "moderate"
        return "weak"

    async def _fetch_news(self, symbol: str) -> List[Dict[str, str]]:
        """Fetch recent news articles about a symbol."""
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": symbol,
            "apiKey": settings.api_key_news,
            "language": "en",
            "sortBy": "publishedAt",
            "pageSize": 10
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                return data.get("articles", [])

    def _aggregate_sentiments(self, sentiments: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not sentiments:
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "error": "No sentiments to analyze",
                "timestamp": datetime.now().isoformat()
            }

        sentiment_counts = {"bullish": 0, "bearish": 0, "neutral": 0}
        confidence_sum = 0.0
        recent_articles = []

        for s in sentiments:
            sentiment_counts[s["sentiment"]] += 1
            confidence_sum += s["confidence"]
            if "source" in s:
                recent_articles.append({
                    "source": s["source"],
                    "sentiment": s["sentiment"],
                    "confidence": s["confidence"],
                    "published_at": s["published_at"]
                })

        dominant_sentiment = max(sentiment_counts.items(), key=lambda x: x[1])[0]
        avg_confidence = confidence_sum / len(sentiments)

        return {
            "sentiment": dominant_sentiment,
            "confidence": avg_confidence,
            "sample_size": len(sentiments),
            "sentiment_distribution": sentiment_counts,
            "recent_articles": recent_articles[:5],  # Include 5 most recent analyzed articles
            "last_updated": datetime.now().isoformat()
        }

    def _get_from_cache(self, key: str) -> Dict[str, Any]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.cache_duration:
                return data
            del self.cache[key]
        return None

    def _add_to_cache(self, key: str, data: Dict[str, Any]):
        self.cache[key] = (data, datetime.now())