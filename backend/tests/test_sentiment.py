import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

def test_get_sentiment(test_client: TestClient, mocker):
    """Test sentiment analysis endpoint."""
    # Mock the sentiment analysis service
    mock_sentiment = {
        "sentiment": "positive",
        "score": 0.8,
        "confidence": 0.9
    }
    
    mocker.patch(
        "src.services.sentiment_analysis.SentimentAnalysisService.analyze_sentiment",
        return_value=mock_sentiment
    )
    
    response = test_client.get("/api/sentiment/AAPL")
    assert response.status_code == 200
    data = response.json()
    
    assert "sentiment" in data
    assert "score" in data
    assert data["sentiment"] in ["positive", "negative", "neutral"]
    assert isinstance(data["score"], float)
