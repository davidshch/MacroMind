"""Core sentiment analysis pipeline using FinBERT."""

from transformers import pipeline
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class SentimentPipeline:
    """Sentiment analysis pipeline optimized for financial text."""
    
    def __init__(self):
        self.analyzer = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            return_all_scores=True
        )
        
    async def analyze(self, text: str) -> Dict[str, Any]:
        """Get sentiment with confidence scores."""
        try:
            results = self.analyzer(text)[0]
            sentiment, confidence = "neutral", 0.0
            
            for score in results:
                if score["score"] > confidence:
                    sentiment = "bullish" if score["label"] == "positive" else "bearish"
                    confidence = score["score"]
            
            return {
                "sentiment": sentiment,
                "confidence": float(confidence),
                "text": text[:100] + "..." if len(text) > 100 else text
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return {"sentiment": "neutral", "confidence": 0.0, "error": str(e)}
