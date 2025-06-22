"""
Sentiment Analysis API Routes.

This module provides endpoints for analyzing market sentiment using AI models.
It combines multiple data sources including news, social media, and market indicators
to generate comprehensive sentiment analysis for financial instruments.
"""

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.ext.asyncio import AsyncSession
from typing import List, Optional
from datetime import date, datetime

from ...schemas.sentiment import (
    AggregatedSentimentResponse,
    TextAnalysisRequest,
    TextAnalysisResponse,
    SentimentInsightResponse,
    SentimentSpikeResponse
)
from ...services.sentiment_analysis import SentimentAnalysisService
from ...database.database import get_db
from ...services.auth import get_current_active_user
from ...database.models import User
from ...core.dependencies import get_sentiment_service

router = APIRouter(tags=["sentiment"])

@router.get("/{symbol}", response_model=AggregatedSentimentResponse)
async def get_aggregated_sentiment_endpoint(
    symbol: str,
    target_date: Optional[date] = Query(None, description="Target date (YYYY-MM-DD). Defaults to today."),
    sentiment_service: SentimentAnalysisService = Depends(get_sentiment_service)
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
    sentiment_service: SentimentAnalysisService = Depends(get_sentiment_service)
):
    """Analyze sentiment of a provided text snippet using the base analyzer."""
    try:
        result = await sentiment_service.analyze_text(request.text)
        if 'timestamp' not in result:
            result['timestamp'] = datetime.now().isoformat()
        return result
    except HTTPException as he:
        raise he
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.exception(f"Error analyzing text: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze text sentiment.")

@router.get("/{symbol}/insights", response_model=SentimentInsightResponse)
async def get_sentiment_insights_endpoint(
    symbol: str,
    lookback_days: int = Query(7, ge=1, le=30, description="Number of past days of sentiment data to analyze."),
    num_themes: int = Query(3, ge=1, le=5, description="Number of key themes to extract."),
    sentiment_service: SentimentAnalysisService = Depends(get_sentiment_service)
):
    """Get AI-distilled sentiment insights (key themes and summary) for a symbol."""
    try:
        insights_result = await sentiment_service.get_distilled_sentiment_insights(
            symbol=symbol, 
            lookback_days=lookback_days, 
            num_themes=num_themes
        )
        
        if "error" in insights_result:
            raise HTTPException(status_code=400, detail=str(insights_result["error"]))
        
        return SentimentInsightResponse(
            themes=insights_result.get("themes", []),
            summary=insights_result.get("summary", "No summary available."),
            asset_symbol=symbol,
            lookback_days=lookback_days,
            timestamp=datetime.now()
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.exception(f"Error getting sentiment insights for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve sentiment insights.")

@router.get("/{symbol}/spikes", response_model=SentimentSpikeResponse)
async def get_sentiment_spikes(
    symbol: str,
    lookback_days: int = Query(90, ge=30, le=365, description="Historical window for spike detection."),
    min_data_points: int = Query(30, ge=10, description="Minimum data points required."),
    sentiment_service: SentimentAnalysisService = Depends(get_sentiment_service)
):
    """Detect unusual spikes or drops in sentiment for a symbol."""
    try:
        spike_data = await sentiment_service.detect_sentiment_spikes(
            symbol=symbol, 
            lookback_days=lookback_days,
            min_data_points=min_data_points
        )
        return SentimentSpikeResponse.model_validate(spike_data)
    except HTTPException as he:
        raise he
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.exception(f"Unexpected error detecting sentiment spikes for {symbol}: {e}")
        raise HTTPException(status_code=500, detail="Failed to detect sentiment spikes.")

# Potential future endpoints:
# - Get historical aggregated sentiment
# - Get sentiment breakdown by source