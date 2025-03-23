"""
Sentiment Analysis API Routes.

This module provides endpoints for analyzing market sentiment using AI models.
It combines multiple data sources including news, social media, and market indicators
to generate comprehensive sentiment analysis for financial instruments.
"""

from fastapi import APIRouter, HTTPException, Depends
from typing import Dict, Any
from ...services.sentiment_analysis import SentimentAnalysisService
from ...services.auth import get_current_user
from ...database.models import User

router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])

@router.get("/analysis/{symbol}", response_model=Dict[str, Any])
async def get_sentiment(
    symbol: str,
    current_user: User = Depends(get_current_user)
):
    """
    Get market sentiment analysis for a specific symbol.

    Analyzes sentiment from news sources using FinBERT model and returns
    a comprehensive sentiment analysis with confidence scores.

    Args:
        symbol (str): Stock/crypto symbol (e.g., "AAPL", "BTC")
        current_user (User): Authenticated user from JWT token

    Returns:
        dict: Sentiment analysis results
        ```json
        {
            "sentiment": "bullish|bearish|neutral",
            "confidence": 0.85,
            "sample_size": 24,
            "sentiment_distribution": {
                "bullish": 12,
                "bearish": 5,
                "neutral": 7
            },
            "recent_articles": [...]
        }
        ```

    Raises:
        HTTPException(500): Server error during sentiment analysis
        HTTPException(404): Symbol not found
    """
    try:
        service = SentimentAnalysisService()
        sentiment = await service.get_market_sentiment(symbol)
        return sentiment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-text")
async def analyze_text(
    text: str,
    current_user: User = Depends(get_current_user)
):
    try:
        service = SentimentAnalysisService()
        result = await service.analyze_text(text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/combined/{symbol}", response_model=Dict[str, Any])
async def get_combined_sentiment(
    symbol: str,
    current_user: User = Depends(get_current_user)
):
    """Get combined sentiment analysis from all sources."""
    try:
        service = SentimentAnalysisService()
        sentiment = await service.get_combined_sentiment(symbol)
        return sentiment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/social/{symbol}", response_model=Dict[str, Any])
async def get_social_sentiment(
    symbol: str,
    current_user: User = Depends(get_current_user)
):
    """Get sentiment analysis from social media sources."""
    try:
        service = SentimentAnalysisService()
        sentiment = await service.social_service.get_reddit_sentiment(symbol)  # Changed from get_combined_social_sentiment
        return sentiment
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))