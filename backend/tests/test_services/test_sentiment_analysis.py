"""Integration tests for volatility-aware sentiment analysis."""

import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import date, timedelta
import numpy as np

# Import SentimentType
from src.database.models import MarketSentiment, SentimentType 
from src.schemas.sentiment import AggregatedSentimentResponse

# Delay service imports
# from src.services.sentiment_analysis import SentimentAnalysisService
# from src.services.volatility import VolatilityService
# from src.services.social_sentiment import SocialSentimentService

@pytest.fixture
def mock_db_session():
    """Provides a MagicMock for the database session."""
    return MagicMock()

@pytest.fixture
def mock_volatility_service():
    """Provides an AsyncMock for the VolatilityService."""
    mock = AsyncMock()
    # Configure default return values for methods used in SentimentAnalysisService
    mock.calculate_and_predict_volatility.return_value = {
        "symbol": "AAPL",
        "market_condition": "normal",
        "is_high_volatility": False,
        "volatility_10d_percentile": 40.0,
        "current_volatility": 0.15,
        "trend": "stable"
    }
    mock.get_market_condition.return_value = {
        "market_condition": "normal",
        "is_high_volatility": False,
        "volatility_trend": "stable",
        "current_percentile": 40.0
    }
    return mock

@pytest.fixture
def mock_social_service():
    """Provides an AsyncMock for the SocialSentimentService."""
    mock = AsyncMock()
    mock.get_reddit_sentiment.return_value = {
        "sentiment": "neutral",
        "confidence": 0.5,
        # Add other fields returned by get_reddit_sentiment if needed
    }
    return mock

@pytest.fixture
def sentiment_service(mock_db_session, mock_volatility_service, mock_social_service):
    """Provides an instance of SentimentAnalysisService with mocked dependencies."""
    # Import service here to use mocks
    from src.services.sentiment_analysis import SentimentAnalysisService
    from src.services.volatility import VolatilityService
    from src.services.social_sentiment import SocialSentimentService

    # Patch the __init__ of the service or the modules it imports if necessary
    # Here, we patch the instances it creates
    with patch('src.services.sentiment_analysis.VolatilityService', return_value=mock_volatility_service), \
         patch('src.services.sentiment_analysis.SocialSentimentService', return_value=mock_social_service):
        service = SentimentAnalysisService(db=mock_db_session)
        # Ensure the mocked services are attached if needed for direct access in tests
        # service.volatility_service = mock_volatility_service
        # service.social_service = mock_social_service
        yield service # Use yield if setup/teardown is needed, else return

@pytest.mark.asyncio
async def test_volatility_aware_sentiment_weighting(sentiment_service, mock_volatility_service):
    """Test dynamic weight adjustment based on volatility."""
    # Arrange
    symbol = "AAPL"
    target_date = date.today()
    mock_news_sentiment = {"sentiment": "bullish", "confidence": 0.8, "score": 0.8}
    mock_reddit_sentiment = {"sentiment": "neutral", "confidence": 0.5, "score": 0.0}

    # Mock internal methods
    sentiment_service._fetch_news_sentiment = AsyncMock(return_value=mock_news_sentiment)
    sentiment_service.social_service.get_reddit_sentiment = AsyncMock(return_value=mock_reddit_sentiment)
    sentiment_service._normalize_sentiment = MagicMock(side_effect=lambda x: {
        'bullish': 1.0, 'neutral': 0.0, 'bearish': -1.0
    }.get(x.get('sentiment', 'neutral'), 0.0) * x.get('confidence', 1.0))
    sentiment_service._calculate_moving_average = AsyncMock(return_value=0.1)
    sentiment_service._select_benchmark = AsyncMock(return_value="SPY")
    sentiment_service.db.query.return_value.filter.return_value.first.return_value = None # No existing record
    sentiment_service.db.add = MagicMock()
    sentiment_service.db.commit = MagicMock()
    sentiment_service.db.refresh = MagicMock()

    # --- Test Case 1: Normal Volatility --- (using default mock_volatility_service setup)
    # Act
    result_normal = await sentiment_service.get_aggregated_sentiment(symbol, target_date)
    # Assert
    assert result_normal.source_weights['news'] == sentiment_service.base_weights['news'] # 0.6
    assert result_normal.source_weights['reddit'] == sentiment_service.base_weights['reddit'] # 0.4
    # Check score calculation (approximate)
    expected_score_normal = (0.8 * 0.6) + (0.0 * 0.4) # (news_norm * news_weight) + (reddit_norm * reddit_weight)
    assert result_normal.normalized_score == pytest.approx(expected_score_normal, abs=1e-9)
    assert result_normal.market_condition == "normal"

    # --- Test Case 2: High Volatility --- #
    # Arrange - Modify volatility mock for this case
    mock_volatility_service.calculate_and_predict_volatility.return_value = {
        "symbol": "AAPL",
        "market_condition": "highly_volatile",
        "is_high_volatility": True,
        "volatility_10d_percentile": 90.0,
        "current_volatility": 0.35,
        "trend": "increasing"
    }
    mock_volatility_service.get_market_condition.return_value = {
        "market_condition": "highly_volatile",
        "is_high_volatility": True,
        "volatility_trend": "increasing",
        "current_percentile": 90.0
    }
    # Re-create service instance with updated mock OR re-patch if possible
    # Easiest is often to rely on the fixture re-running or modify the existing mock instance

    # Act
    result_high_vol = await sentiment_service.get_aggregated_sentiment(symbol, target_date)

    # Assert
    high_vol_weights = sentiment_service.volatility_adjustments['highly_volatile']
    assert result_high_vol.source_weights['news'] == high_vol_weights['news'] # 0.7
    assert result_high_vol.source_weights['reddit'] == high_vol_weights['reddit'] # 0.3
    expected_score_high = (0.8 * 0.7) + (0.0 * 0.3)
    assert result_high_vol.normalized_score == pytest.approx(expected_score_high, abs=1e-9)
    assert result_high_vol.market_condition == "highly_volatile"

    # --- Test Case 3: Low Volatility --- #
    # Arrange - Modify volatility mock
    mock_volatility_service.calculate_and_predict_volatility.return_value = {
        "symbol": "AAPL",
        "market_condition": "low_volatility",
        "is_high_volatility": False,
        "volatility_10d_percentile": 10.0,
        "current_volatility": 0.05,
        "trend": "decreasing"
    }
    mock_volatility_service.get_market_condition.return_value = {
        "market_condition": "low_volatility",
        "is_high_volatility": False,
        "volatility_trend": "decreasing",
        "current_percentile": 10.0
    }

    # Act
    result_low_vol = await sentiment_service.get_aggregated_sentiment(symbol, target_date)

    # Assert
    low_vol_weights = sentiment_service.volatility_adjustments['low_volatility']
    assert result_low_vol.source_weights['news'] == low_vol_weights['news'] # 0.55
    assert result_low_vol.source_weights['reddit'] == low_vol_weights['reddit'] # 0.45
    expected_score_low = (0.8 * 0.55) + (0.0 * 0.45)
    assert result_low_vol.normalized_score == pytest.approx(expected_score_low, abs=1e-9)
    assert result_low_vol.market_condition == "low_volatility"

    # Verify DB interaction (example for one case)
    sentiment_service.db.add.assert_called_once()
    call_args = sentiment_service.db.add.call_args[0][0]
    assert isinstance(call_args, MarketSentiment)
    assert call_args.symbol == symbol
    assert call_args.date == target_date
    assert call_args.sentiment == SentimentType.BULLISH # Based on final score > threshold
    assert call_args.score == pytest.approx(expected_score_low)
    sentiment_service.db.commit.assert_called_once()
    sentiment_service.db.refresh.assert_called_once()
