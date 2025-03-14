from typing import Dict, Any
from transformers import pipeline
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class BaseSentimentAnalyzer:
    def __init__(self):
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert"
        )
        self.sentiment_mapping = {
            "positive": "bullish",
            "negative": "bearish",
            "neutral": "neutral"
        }

    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Base method for text sentiment analysis."""
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
            raise
