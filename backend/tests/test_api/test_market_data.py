import pytest
from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_get_market_data():
    """Test market data endpoint returns correct structure."""
    response = client.get("/api/market-data/AAPL")
    assert response.status_code == 200
    data = response.json()
    
    # Validate response structure
    assert "symbol" in data
    assert "price" in data
    assert "change" in data
    assert isinstance(data["price"], (int, float))

def test_get_market_sentiment():
    """Test market sentiment endpoint."""
    response = client.get("/api/sentiment/AAPL")
    assert response.status_code == 200
    data = response.json()
    
    # Validate sentiment structure
    assert "sentiment_score" in data
    assert "sentiment_label" in data
    assert data["sentiment_label"] in ["bullish", "bearish", "neutral"]

@pytest.mark.asyncio
async def test_websocket_connection():
    """Test WebSocket connection and data streaming."""
    with client.websocket_connect("/ws/AAPL") as websocket:
        data = websocket.receive_json()
        assert "symbol" in data
        assert "price" in data