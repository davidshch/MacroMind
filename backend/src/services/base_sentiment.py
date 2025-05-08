"""BaseModel for sentiment analysis services.

This module implements the base sentiment analyzer with FinBERT pipeline.
"""

from typing import Dict, Any, Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
from datetime import datetime
import logging
import torch
import os
from ..config import get_settings

settings = get_settings()
logger = logging.getLogger(__name__)

class BaseSentimentAnalyzer:
    """Base sentiment analyzer with FinBERT pipeline."""

    _instance = None
    _sentiment_analyzer = None
    _model = None
    _tokenizer = None

    def __new__(cls):
        """Implement as singleton to avoid loading the model multiple times."""
        if cls._instance is None:
            cls._instance = super(BaseSentimentAnalyzer, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """Initialize the sentiment analyzer."""
        if not hasattr(self, "_initialized") or not self._initialized:
            self.sentiment_mapping = {
                "positive": "bullish",
                "negative": "bearish",
                "neutral": "neutral"
            }
            # Don't load model on init - lazy load it when needed
            self._initialized = True
            logger.info("BaseSentimentAnalyzer initialized as singleton")

    @property
    def sentiment_analyzer(self):
        """Lazy loading of the sentiment analysis pipeline."""
        if BaseSentimentAnalyzer._sentiment_analyzer is None:
            try:
                logger.info("Loading FinBERT model for sentiment analysis")
                # If running in test mode, use a lightweight dummy model
                if os.getenv('TESTING') == 'TRUE':
                    logger.info("Using mock sentiment analyzer for testing")
                    return self._get_mock_analyzer()

                # Set up the model with better memory management
                model_name = "ProsusAI/finbert"
                
                # Load models with safe fallbacks
                if BaseSentimentAnalyzer._model is None:
                    try:
                        BaseSentimentAnalyzer._model = AutoModelForSequenceClassification.from_pretrained(model_name)
                    except Exception as e:
                        logger.error(f"Error loading FinBERT model: {str(e)}")
                        return self._get_mock_analyzer()
                
                if BaseSentimentAnalyzer._tokenizer is None:
                    try:
                        BaseSentimentAnalyzer._tokenizer = AutoTokenizer.from_pretrained(model_name)
                    except Exception as e:
                        logger.error(f"Error loading FinBERT tokenizer: {str(e)}")
                        return self._get_mock_analyzer()
                
                # Use CPU by default for testing environment
                device = -1  # CPU
                if torch.cuda.is_available() and not os.getenv('TESTING') == 'TRUE':
                    device = 0  # GPU if available and not in testing mode
                
                BaseSentimentAnalyzer._sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model=BaseSentimentAnalyzer._model,
                    tokenizer=BaseSentimentAnalyzer._tokenizer,
                    device=device
                )
                logger.info("FinBERT model loaded successfully")
            except Exception as e:
                logger.error(f"Error initializing sentiment analyzer: {str(e)}")
                return self._get_mock_analyzer()
                
        return BaseSentimentAnalyzer._sentiment_analyzer

    def _get_mock_analyzer(self):
        """Return a mock analyzer for testing or when model fails to load."""
        # This is a callable that mimics the pipeline's behavior
        def mock_analyzer(text):
            import random
            # Determine sentiment based on keywords if possible
            text_lower = text.lower()
            result = {"label": "neutral", "score": 0.6}
            
            # Simple rule-based classification for predictable test results
            if any(word in text_lower for word in ["growth", "increase", "gain", "up", "bullish", "positive"]):
                result = {"label": "positive", "score": 0.8}
            elif any(word in text_lower for word in ["drop", "decline", "loss", "down", "bearish", "negative"]):
                result = {"label": "negative", "score": 0.8}
            else:
                # Add some randomness for neutral text
                sentiments = ["neutral", "positive", "negative"]
                weights = [0.6, 0.2, 0.2]
                result["label"] = random.choices(sentiments, weights=weights)[0]
                result["score"] = 0.5 + random.random() * 0.3
            
            return [result]  # Match pipeline output format
            
        return mock_analyzer

    async def analyze_text(self, text: str) -> Dict[str, Any]:
        """Analyze text sentiment and map to market context.
        
        Args:
            text: The text to analyze
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            if not text:
                return {
                    "sentiment": "neutral",
                    "confidence": 0.0,
                    "timestamp": datetime.now().isoformat(),
                    "original_text": ""
                }

            # Get the active analyzer (real or mock)
            analyzer = self.sentiment_analyzer
            
            # Analyze text
            result = analyzer(text)[0]
            
            # Map to market terminology
            mapped_sentiment = self.sentiment_mapping.get(result["label"], "neutral")
            
            return {
                "sentiment": mapped_sentiment,
                "confidence": float(result["score"]),
                "timestamp": datetime.now().isoformat(),
                "original_text": text[:100] + "..." if len(text) > 100 else text
            }
        except Exception as e:
            logger.error(f"Error analyzing text: {str(e)}")
            # Return neutral sentiment on error for robustness
            return {
                "sentiment": "neutral",
                "confidence": 0.0,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "original_text": text[:100] + "..." if len(text) > 100 else text
            }

    def __del__(self):
        """Clean up resources when the analyzer is destroyed."""
        try:
            # Clear memory when instance is destroyed
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            logger.error(f"Error cleaning up resources: {str(e)}")