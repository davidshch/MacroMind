import logging
from transformers import pipeline
from typing import Dict, Any, List
import aiohttp
from ..config import get_settings
from datetime import datetime, timedelta
from fastapi import HTTPException

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

settings = get_settings()

class SentimentAnalysisService:
    def __init__(self):
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )
        self.cache = {}
        self.cache_duration = timedelta(hours=1)
        self.sentiment_mapping = {
            "positive": "bullish",
            "negative": "bearish",
            "neutral": "neutral"
        }

    async def analyze_text(self, text: str) -> Dict[str, Any]:
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