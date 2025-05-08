"""
Sentiment Analysis API Routes.

This module provides endpoints for analyzing market sentiment using AI models.
It combines multiple data sources including news, social media, and market indicators
to generate comprehensive sentiment analysis for financial instruments.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import date

from ..schemas.sentiment import (
    AggregatedSentimentResponse,
    TextAnalysisRequest,
    TextAnalysisResponse
)
from ..services.sentiment_analysis import SentimentAnalysisService, get_sentiment_analysis_service
from ..database.database import get_db
from ..services.auth import get_current_active_user
from ..database.models import User

router = APIRouter(prefix="/api/sentiment", tags=["sentiment"])

@router.get("/{symbol}", response_model=AggregatedSentimentResponse)
async def get_aggregated_sentiment_endpoint(
    symbol: str,
    target_date: Optional[date] = Query(None, description="Target date (YYYY-MM-DD). Defaults to today."),
    sentiment_service: SentimentAnalysisService = Depends(get_sentiment_analysis_service),
    current_user: User = Depends(get_current_active_user)
):
    """Get aggregated market sentiment for a symbol on a specific date."""
    if target_date is None:
        target_date = date.today()
    try:
        result = await sentiment_service.get_aggregated_sentiment(symbol=symbol, target_date=target_date)
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.exception(f"Error getting aggregated sentiment for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve aggregated sentiment.")

@router.post("/analyze-text", response_model=TextAnalysisResponse)
async def analyze_text_endpoint(
    request: TextAnalysisRequest,
    current_user: User = Depends(get_current_active_user),
    sentiment_service: SentimentAnalysisService = Depends(get_sentiment_analysis_service)
):
    """Analyze sentiment of a provided text snippet using the base analyzer."""
    try:
        result = await sentiment_service.analyze_text(request.text)
        if 'timestamp' not in result:
             from datetime import datetime
             result['timestamp'] = datetime.now().isoformat()
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.exception(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze text sentiment.")

# Potential future endpoints:
# - Get historical aggregated sentiment
# - Get sentiment breakdown by source