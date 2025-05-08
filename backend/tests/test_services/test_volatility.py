"""Tests for volatility prediction service and endpoints."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import asyncio # Added for sleep

# Delay import of the service
# from src.services.volatility import VolatilityService
from src.schemas.volatility import VolatilityResponse, VolatilityRegime

@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing."""
    dates = pd.date_range(end=datetime.now(), periods=60)
    prices = pd.Series(np.random.normal(100, 2, 60)).cumsum()
    
    return [
        {
            "date": date.strftime("%Y-%m-%d"),
            "close": price,
            "high": price * 1.02,
            "low": price * 0.98,
            "volume": int(np.random.normal(1000000, 100000))
        }
        for date, price in zip(dates, prices)
    ]

@pytest.mark.asyncio
async def test_volatility_calculation(sample_market_data):
    """Test basic volatility calculation functionality."""
    # Import service here
    from src.services.volatility import VolatilityService
    service = VolatilityService()
    
    # Mock market data service response
    service.market_data.get_historical_prices = AsyncMock(return_value=sample_market_data)
    
    result = await service.calculate_and_predict_volatility("AAPL")
    
    assert isinstance(result, dict)
    assert "current_volatility" in result
    assert "predicted_volatility" in result
    assert "market_conditions" in result
    assert result["symbol"] == "AAPL"
    assert isinstance(result["historical_volatility_annualized"], float)
    assert isinstance(result["is_high_volatility"], bool)
    assert result["trend"] in ["increasing", "decreasing"]

@pytest.mark.asyncio
async def test_market_condition_assessment():
    """Test market condition classification."""
    # Import service here
    from src.services.volatility import VolatilityService
    service = VolatilityService()
    
    # Test highly volatile conditions
    current_vol = {
        "is_high": True,
        "percentile": 90.0,
        "current": 0.25,
        "annualized": 0.35
    }
    predictions = {
        "trend": "increasing",
        "mean": 0.3,
        "low": 0.2,
        "high": 0.4,
        "confidence": 0.8
    }
    
    condition = service._assess_market_condition(current_vol, predictions)
    assert condition == "highly_volatile"
    
    # Test normal conditions
    current_vol["is_high"] = False
    current_vol["percentile"] = 50.0
    predictions["trend"] = "decreasing"
    
    condition = service._assess_market_condition(current_vol, predictions)
    assert condition == "normal"

@pytest.mark.asyncio
async def test_volatility_prediction_features(sample_market_data):
    """Test feature preparation for volatility prediction."""
    # Import service here
    from src.services.volatility import VolatilityService
    service = VolatilityService()
    
    df = service._prepare_features(sample_market_data)
    
    # Check if all required features are present
    required_features = ['returns', 'vol_5d', 'vol_10d', 'vol_30d', 'rsi', 'atr']
    for feature in required_features:
        assert feature in df.columns
    
    # Check if features have expected properties
    assert not df['returns'].isna().any()
    assert df['vol_5d'].min() >= 0  # Volatility should be non-negative
    assert df['rsi'].between(0, 100).all()  # RSI should be between 0 and 100

@pytest.mark.asyncio
async def test_cache_functionality():
    """Test caching behavior of volatility service."""
    # Import service here
    from src.services.volatility import VolatilityService
    service = VolatilityService()
    
    # Add test data to cache
    test_data = {"test": "data"}
    service._add_to_cache("test_key", test_data)
    
    # Verify cache retrieval works
    cached = service._get_from_cache("test_key")
    assert cached == test_data
    
    # Verify cache expiration works
    service.cache_duration = timedelta(milliseconds=1)
    await asyncio.sleep(0.01)
    expired = service._get_from_cache("test_key")
    assert expired is None

@pytest.fixture
def auth_headers():
    """Generate mock authentication headers."""
    return {"Authorization": "Bearer test-token"}

@pytest.mark.asyncio
async def test_volatility_api_endpoint(test_client, auth_headers):
    """Test the volatility prediction API endpoint."""
    # No service import needed here as it tests the API endpoint
    response = await test_client.get(
        "/api/volatility/AAPL",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert "symbol" in data
    assert "current_volatility" in data
    assert "predicted_volatility" in data
    assert "market_conditions" in data
    assert data["symbol"] == "AAPL"
    
    # Verify prediction range
    assert "prediction_range" in data
    assert data["prediction_range"]["low"] <= data["prediction_range"]["high"]
