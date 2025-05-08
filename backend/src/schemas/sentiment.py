from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import date, datetime

from ..database.models import SentimentType # Corrected import path

class SentimentBase(BaseModel):
    symbol: str
    date: date
    sentiment: SentimentType
    score: float # Normalized score (-1 to 1)
    avg_daily_score: Optional[float] = None
    moving_avg_7d: Optional[float] = None
    news_score: Optional[float] = None
    reddit_score: Optional[float] = None
    benchmark: Optional[str] = None
    timestamp: datetime

class SentimentCreate(SentimentBase):
    pass

class SentimentResponse(SentimentBase):
    id: int

    class Config:
        from_attributes = True

class SentimentAnalysisResult(BaseModel):
    sentiment: str
    confidence: float
    timestamp: datetime
    original_text: Optional[str] = None
    source: Optional[str] = None
    published_at: Optional[datetime] = None

class TextAnalysisRequest(BaseModel):
    text: str

class TextAnalysisResponse(BaseModel):
    sentiment: str
    confidence: float
    timestamp: datetime

class VolatilityContext(BaseModel):
    """Schema for volatility context within sentiment response."""
    level: float
    is_high: bool
    trend: str

class AggregatedSentimentResponse(BaseModel):
    id: int
    symbol: str
    date: date
    overall_sentiment: SentimentType
    normalized_score: float = Field(..., description="Overall aggregated score, normalized -1 to 1")
    avg_daily_score: Optional[float] = None # Kept for potential future use
    moving_avg_7d: Optional[float] = None
    benchmark: Optional[str] = None
    news_sentiment: Optional[Dict[str, Any]] = None # Raw news sentiment result
    reddit_sentiment: Optional[Dict[str, Any]] = None # Using Dict for Reddit's complex structure
    market_condition: str = Field(..., description="Market condition assessment from VolatilityService") # Added
    volatility_context: VolatilityContext = Field(..., description="Detailed volatility context") # Added
    source_weights: Dict[str, float] = Field(..., description="Weights used for source aggregation") # Added
    timestamp: datetime

    class Config:
        from_attributes = True
        use_enum_values = True # Ensure enum values are used in response
