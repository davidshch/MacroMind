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