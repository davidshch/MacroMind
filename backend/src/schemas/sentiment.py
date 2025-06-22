from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import date, datetime

from ..database.models import SentimentType # Corrected import path

class SentimentBase(BaseModel):
    symbol: str
    date: date
    sentiment: SentimentType
    score: float # Normalized score (-1 to 1)
    avg_daily_score: Optional[float] = None
    moving_avg_7d: Optional[float] = None
    news_score: Optional[float] = None # Normalized news score
    reddit_score: Optional[float] = None # Normalized Reddit score
    benchmark: Optional[str] = None
    timestamp: datetime

class SentimentCreate(SentimentBase):
    pass

class SentimentResponse(SentimentBase): # This seems like a general DB model response
    id: int

    class Config:
        from_attributes = True
        use_enum_values = True


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
    overall_sentiment: SentimentType # Enum: e.g., "bullish"
    normalized_score: float = Field(..., description="Overall aggregated score, normalized -1 to 1")
    avg_daily_score: Optional[float] = Field(None, description="Average daily score, potentially same as normalized_score or for future use")
    moving_avg_7d: Optional[float] = Field(None, description="7-day moving average of the normalized_score")
    benchmark: Optional[str] = Field(None, description="Dynamically selected benchmark (e.g., SPY, QQQ, RSP")
    
    # Option 1: Keep raw source data if needed by frontend
    news_sentiment_details: Optional[Dict[str, Any]] = Field(None, description="Raw details from news sentiment analysis, if available")
    reddit_sentiment_details: Optional[Dict[str, Any]] = Field(None, description="Raw details from Reddit sentiment analysis, if available")

    # Option 2: Or just include the normalized scores from each source
    news_sentiment_score: Optional[float] = Field(None, description="Normalized score from news sources (-1 to 1)")
    reddit_sentiment_score: Optional[float] = Field(None, description="Normalized score from Reddit sources (-1 to 1)")
    
    market_condition: str = Field(..., description="Market condition assessment from VolatilityService")
    volatility_context: VolatilityContext = Field(..., description="Detailed volatility context")
    source_weights: Dict[str, float] = Field(..., description="Weights used for source aggregation")
    timestamp: datetime
    
    # Additional context fields
    data_quality_score: Optional[float] = Field(None, description="Score indicating quality of available data (0-1)")
    sentiment_strength: Optional[float] = Field(None, description="Absolute strength of sentiment regardless of direction (0-1)")
    sentiment_confidence: Optional[float] = Field(None, description="Overall confidence in the sentiment analysis (0-1)")
    data_sources_available: Optional[Dict[str, bool]] = Field(None, description="Which data sources contributed to the analysis")

    class Config:
        from_attributes = True # Allows Pydantic to map ORM model fields
        use_enum_values = True # Ensures enum values (strings) are used in the response

class SentimentInsightResponse(BaseModel):
    themes: List[str] = Field(..., description="List of key sentiment themes identified by the LLM.")
    summary: str = Field(..., description="A brief summary from the LLM about the sentiment drivers.")
    asset_symbol: str = Field(..., description="The asset symbol for which insights were generated.")
    lookback_days: int = Field(..., description="The number of days of sentiment data considered.")
    timestamp: datetime = Field(default_factory=datetime.now, description="Timestamp of when the insights were generated.")

    class Config:
        from_attributes = True

class SentimentSpikeData(BaseModel):
    date: date
    actual_score: float
    expected_score_upper: float
    expected_score_lower: float
    is_spike: bool
    # Optional: Add fields for correlated news/events if implementing that part
    # correlated_news_headlines: List[str] = []

class SentimentSpikeResponse(BaseModel):
    symbol: str
    time_period_days: int
    spikes_detected: List[SentimentSpikeData]
    message: Optional[str] = None
