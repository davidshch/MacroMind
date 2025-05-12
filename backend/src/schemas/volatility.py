"""Pydantic schemas for volatility-related data structures."""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime, date
from enum import Enum

class VolatilityRegime(str, Enum):
    CRISIS = "crisis"
    HIGH_INCREASING = "high_increasing"
    HIGH_STABLE = "high_stable"
    NORMAL = "normal"
    LOW_STABLE = "low_stable"
    LOW_COMPRESSION = "low_compression"

class PredictionRange(BaseModel):
    low: float = Field(..., description="Lower bound of volatility prediction")
    high: float = Field(..., description="Upper bound of volatility prediction")

class VolatilityMetadata(BaseModel):
    model_version: str = Field(..., description="Version of the volatility model")
    features_used: List[str] = Field(..., description="Features used in prediction")
    last_updated: datetime = Field(..., description="Timestamp of last update")

class VolatilityResponse(BaseModel):
    """Enhanced response model for volatility predictions."""
    symbol: str = Field(..., description="Trading symbol")
    timestamp: datetime = Field(..., description="Prediction timestamp")
    current_volatility: float = Field(..., description="Current volatility value")
    historical_volatility_annualized: float = Field(..., 
        description="Annualized historical volatility")
    volatility_10d_percentile: float = Field(..., 
        description="10-day volatility percentile")
    predicted_volatility: float = Field(..., 
        description="Mean predicted volatility")
    prediction_range: PredictionRange = Field(..., 
        description="Prediction confidence interval")
    market_conditions: str = Field(..., 
        description="Current market condition assessment")
    volatility_regime: VolatilityRegime = Field(..., 
        description="Current volatility regime")
    is_high_volatility: bool = Field(..., 
        description="Flag for high volatility conditions")
    trend: str = Field(..., description="Volatility trend direction")
    confidence_score: float = Field(..., ge=0, le=1, 
        description="Model confidence score")
    metadata: VolatilityMetadata = Field(..., 
        description="Prediction metadata")

class VolatilityRequest(BaseModel):
    """Request model for volatility predictions."""
    symbol: str = Field(..., description="Trading symbol to analyze")
    lookback_days: int = Field(30, ge=5, le=252, 
        description="Days of historical data to use")
    prediction_days: int = Field(5, ge=1, le=30, 
        description="Number of days to predict ahead")

class HistoricalVolatility(BaseModel):
    """Historical volatility data points."""
    date: datetime
    value: float
    regime: VolatilityRegime
    percentile: float

class TrainVolatilityModelRequest(BaseModel):
    """Request model for training a volatility model for a specific symbol."""
    symbol: str = Field(..., description="The stock/asset symbol for which to train the model.")
    training_start_date_str: str = Field(..., description="Start date for training data (YYYY-MM-DD).")
    training_end_date_str: str = Field(..., description="End date for training data (YYYY-MM-DD).")
    future_vol_period_days: int = Field(5, ge=1, le=30, description="Number of future days to predict volatility for (target variable).")

class TrainVolatilityModelResponse(BaseModel):
    """Response model for the volatility model training status."""
    status: str = Field(..., description="Status of the training job (e.g., 'success', 'error').")
    message: str = Field(..., description="A message detailing the outcome of the training.")
    model_path: Optional[str] = Field(None, description="Path where the trained model is saved, if successful.")
    features_used: Optional[List[str]] = Field(None, description="List of features used for training, if successful.")