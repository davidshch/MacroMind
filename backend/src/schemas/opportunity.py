"""
Pydantic schemas for Market Opportunity Highlighter.
"""
from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

from .volatility import VolatilityForecast # Assuming a schema for volatility forecast details
from .sentiment import AggregatedSentimentResponse # Re-using for sentiment details

class MarketOpportunityHighlight(BaseModel):
    symbol: str
    timestamp: datetime
    reason: str # e.g., "High Volatility & Strong Bullish Sentiment"
    volatility_forecast: Optional[VolatilityForecast] = None # Detailed forecast if available
    current_volatility: Optional[float] = None
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    # Consider adding links to relevant data or charts in the future

class MarketOpportunityResponse(BaseModel):
    generated_at: datetime
    highlights: List[MarketOpportunityHighlight]
    assets_checked: int
    message: Optional[str] = None
