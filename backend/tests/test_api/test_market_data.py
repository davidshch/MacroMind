import pytest
from fastapi.testclient import TestClient

def test_get_market_data(test_client: TestClient):
    """Test market data endpoint returns correct structure."""
    response = test_client.get("/api/market/stock/AAPL")
    assert response.status_code == 200
    data = response.json()
    
    # Validate response structure
    assert "symbol" in data
    assert "price" in data
    assert "change" in data
    assert isinstance(data["price"], (int, float))

def test_get_market_sentiment(test_client: TestClient):
    """Test market sentiment endpoint."""
    response = test_client.get("/api/sentiment/analysis/AAPL")
    assert response.status_code == 200
    data = response.json()
    
    # Validate sentiment structure
    assert "sentiment_score" in data
    assert "sentiment_label" in data
    assert data["sentiment_label"] in ["bullish", "bearish", "neutral"]

@pytest.mark.asyncio
async def test_websocket_connection(test_client: TestClient):
    """Test WebSocket connection and data streaming."""
    pass  # Placeholder for actual websocket test logic