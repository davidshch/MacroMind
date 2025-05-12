"""
Core dependency providers for the application.
"""
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Optional

from ..database.database import get_db
from ..services.ml.model_factory import MLModelFactory
from ..services.market_data import MarketDataService
from ..services.sentiment_analysis import SentimentAnalysisService
from ..services.volatility import VolatilityService
from ..services.economic_calendar import EconomicCalendarService

# --- ML Model Factory --- #
_ml_model_factory_instance: Optional[MLModelFactory] = None

def get_ml_model_factory() -> MLModelFactory:
    global _ml_model_factory_instance
    if _ml_model_factory_instance is None:
        _ml_model_factory_instance = MLModelFactory()
    return _ml_model_factory_instance

# --- Market Data Service --- #
_market_data_service_instance: Optional[MarketDataService] = None

def get_market_data_service() -> MarketDataService:
    # Simple singleton for MarketDataService as it doesn't have complex deps like DB
    global _market_data_service_instance
    if _market_data_service_instance is None:
        _market_data_service_instance = MarketDataService()
    return _market_data_service_instance

# --- Volatility Service --- #
async def get_volatility_service(
    db: AsyncSession = Depends(get_db),
    market_data_service: MarketDataService = Depends(get_market_data_service),
    ml_model_factory: MLModelFactory = Depends(get_ml_model_factory)
) -> VolatilityService:
    return VolatilityService(
        db=db, 
        market_data_service=market_data_service, 
        ml_model_factory=ml_model_factory
    )

# --- Sentiment Analysis Service --- #
async def get_sentiment_service(
    db: AsyncSession = Depends(get_db),
    market_data_service: MarketDataService = Depends(get_market_data_service),
    ml_model_factory: MLModelFactory = Depends(get_ml_model_factory),
    volatility_service: VolatilityService = Depends(get_volatility_service) # Now correctly injected
) -> SentimentAnalysisService:
    return SentimentAnalysisService(
        db=db, 
        market_data_service=market_data_service, 
        ml_model_factory=ml_model_factory,
        volatility_service=volatility_service # Pass injected service
    )

# EconomicCalendarService, AlertService, OpportunityHighlightService have providers in their own files.
# This is acceptable, or they could be moved here for full centralization.
