"""Service provider functions for FastAPI dependency injection."""

from typing import Optional

from ..services.market_data import MarketDataService
from ..services.ml.model_factory import MLModelFactory

# Singleton instances
_market_data_service: Optional[MarketDataService] = None
_ml_model_factory: Optional[MLModelFactory] = None

def get_market_data_service() -> MarketDataService:
    """Get or create MarketDataService singleton instance."""
    global _market_data_service
    if _market_data_service is None:
        _market_data_service = MarketDataService()
    return _market_data_service

def get_ml_model_factory() -> MLModelFactory:
    """Get or create MLModelFactory singleton instance."""
    global _ml_model_factory
    if _ml_model_factory is None:
        _ml_model_factory = MLModelFactory()
    return _ml_model_factory
