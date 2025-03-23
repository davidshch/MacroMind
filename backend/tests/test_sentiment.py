import pytest
from fastapi.testclient import TestClient
from datetime import datetime

def test_get_sentiment(client: TestClient, mocker):
    # Mock the sentiment analysis service
    mocker.patch(
        "src.services.sentiment_analysis.SentimentAnalysisService.get_market_sentiment",
        return_value={
            "sentiment": "bullish",
            "confidence": 0.85,
            "timestamp": datetime.now().isoformat()
        }
    )
    
    response = client.get("/api/sentiment/analysis/AAPL")
    assert response.status_code == 200
    data = response.json()
    assert "sentiment" in data
    assert "confidence" in data
    assert data["sentiment"] in ["bullish", "bearish", "neutral"]
