
import pytest
from fastapi import HTTPException
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import date, datetime, timedelta

from src.database.models import SectorFundamental
from src.schemas.sector import SectorFundamentalResponse
from src.config import Settings # Import Settings

# Delay service imports
from src.services.sector_fundamentals import SectorFundamentalsService, SECTOR_ETF_MAP
# from src.services.market_data_services import AlphaVantageService, FinnhubService

@pytest.fixture
def mock_db_session():
    """Provides a MagicMock for the database session."""
    return MagicMock()

@pytest.fixture
def mock_alpha_vantage_service():
    """Provides an AsyncMock for the AlphaVantageService."""
    mock = AsyncMock()
    # Configure return values if methods are called
    mock.get_sector_performance.return_value = {"some": "data"} # Example
    return mock

@pytest.fixture
def mock_finnhub_service():
    """Provides an AsyncMock for the FinnhubService."""
    mock = AsyncMock()
    # Configure return values for methods called
    mock.get_company_metrics.return_value = {
        'metric': {
            'peNormalizedAnnual': 15.5,
            'pbAnnual': 2.1,
            'epsGrowthTTMYoy': 0.12
        }
    }
    return mock

@pytest.fixture
def mock_settings(): # Add mock settings fixture
    """Provides mock settings."""
    return Settings(
        database_url="sqlite+aiosqlite:///./test.db", # Example test DB URL
        api_key_alpha_vantage="test_av_key",
        finnhub_api_key="test_fh_key",
        jwt_secret="test_secret",
        jwt_algorithm="HS256",
        access_token_expire_minutes=30,
        reddit_client_id="test_reddit_id",
        reddit_client_secret="test_reddit_secret",
        use_demo_data=True
    )

@pytest.fixture
def sector_service(mock_db_session, mock_alpha_vantage_service, mock_finnhub_service, mock_settings):
    """Provides an instance of SectorFundamentalsService with mocked dependencies."""
    # Import service here
    from src.services.sector_fundamentals import SectorFundamentalsService
    from src.services.market_data_services import AlphaVantageService, FinnhubService
    from src.config import get_settings

    # Patch the dependencies WITHIN the sector_fundamentals module where they are IMPORTED/USED
    with patch('src.services.sector_fundamentals.get_settings', return_value=mock_settings), \
         patch('src.services.sector_fundamentals.AlphaVantageService', return_value=mock_alpha_vantage_service), \
         patch('src.services.sector_fundamentals.FinnhubService', return_value=mock_finnhub_service):
        # Also patch the finnhub.Client call inside the service's __init__
        with patch('finnhub.Client') as mock_finnhub_client_constructor:
            # Configure the instance returned by the constructor if needed
            # mock_finnhub_instance = mock_finnhub_client_constructor.return_value
            # mock_finnhub_instance.company_basic_financials.return_value = ... # Example

            service = SectorFundamentalsService(db=mock_db_session)
            # Manually assign mocked services if __init__ doesn't use the patched classes directly
            # service.alpha_vantage = mock_alpha_vantage_service
            # service.finnhub = mock_finnhub_service
            yield service

# --- Test Cases ---

@pytest.mark.asyncio
async def test_get_sector_fundamentals_new_record(sector_service, mock_db_session, mock_finnhub_service):
    """Test fetching fundamentals when not in DB."""
    # Arrange
    from src.services.sector_fundamentals import SECTOR_ETF_MAP
    sector_name = "Technology"
    etf_symbol = SECTOR_ETF_MAP[sector_name]
    target_date = date.today()

    mock_db_session.query.return_value.filter.return_value.first.return_value = None # Simulate not found in DB

    # Mock the Finnhub client call within the service method
    # This assumes the service uses self.finnhub.get_company_metrics or similar
    # If it uses finnhub.Client directly, the patch in the fixture should handle it.
    # Let's refine the mock for the specific call used in _fetch_and_store...
    # The service uses asyncio.to_thread(self.finnhub_client.company_basic_financials, ...)
    # We need to ensure the mock_finnhub_client_constructor patch covers this.

    # Mock the Finnhub data that _fetch_and_store_sector_fundamentals expects
    finnhub_data = {
        'metric': {
            'peNormalizedAnnual': 25.0,
            'pbAnnual': 4.5,
            'epsGrowthTTMYoy': 0.15
        }
    }
    # Since the actual call is inside asyncio.to_thread, mocking the service's finnhub instance method might be tricky.
    # Let's mock the `asyncio.to_thread` call itself for this specific test.
    with patch('asyncio.to_thread', new_callable=AsyncMock) as mock_to_thread:
        mock_to_thread.return_value = finnhub_data

        # Mock DB operations for adding the new record
        mock_db_session.add = MagicMock()
        mock_db_session.commit = MagicMock()
        mock_db_session.refresh = MagicMock(side_effect=lambda x: setattr(x, 'id', 1)) # Simulate refresh setting an ID

        # Act
        result = await sector_service.get_sector_fundamentals(sector_name, target_date)

        # Assert
        assert result is not None
        assert isinstance(result, SectorFundamentalResponse)
        assert result.sector_name == sector_name
        assert result.date == target_date
        assert result.pe_ratio == 25.0
        assert result.pb_ratio == 4.5
        assert result.earnings_growth == 0.15

        # Verify Finnhub call was attempted via asyncio.to_thread
        mock_to_thread.assert_called_once()
        # Verify DB add/commit/refresh
        mock_db_session.add.assert_called_once()
        mock_db_session.commit.assert_called_once()
        mock_db_session.refresh.assert_called_once()

@pytest.mark.asyncio
async def test_get_sector_fundamentals_existing_record(sector_service, mock_db_session, mock_alpha_vantage_service, mock_finnhub_service):
    """Test fetching fundamentals when a record for the date already exists."""
    # Arrange
    sector_name = "Healthcare"
    target_date = date.today()
    existing_record = SectorFundamental(
        id=456,
        sector_name=sector_name,
        date=target_date,
        pe_ratio=20.0,
        pb_ratio=3.0,
        earnings_growth=0.08,
        timestamp=datetime.now()
    )
    mock_db_session.query.return_value.filter.return_value.first.return_value = existing_record

    # Act
    result = await sector_service.get_sector_fundamentals(sector_name, target_date)

    # Assert
    assert isinstance(result, SectorFundamentalResponse)
    assert result.id == 456
    assert result.sector_name == sector_name
    assert result.date == target_date
    assert result.pe_ratio == 20.0
    assert result.pb_ratio == 3.0
    assert result.earnings_growth == 0.08

    # Verify NO external API calls were made
    mock_alpha_vantage_service.get_overview.assert_not_called()
    mock_finnhub_service.get_company_basic_financials.assert_not_called()

    # Verify NO DB write operations were made
    mock_db_session.add.assert_not_called()
    mock_db_session.commit.assert_not_called()
    mock_db_session.refresh.assert_not_called()

@pytest.mark.asyncio
async def test_get_sector_fundamentals_invalid_sector(sector_service):
    """Test fetching fundamentals for an unsupported sector."""
    # Arrange
    sector_name = "Underwater Basket Weaving"
    target_date = date.today()

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        await sector_service.get_sector_fundamentals(sector_name, target_date)
    assert exc_info.value.status_code == 404
    assert f"Unsupported sector: {sector_name}" in exc_info.value.detail

@pytest.mark.asyncio
async def test_update_all_sector_fundamentals(sector_service, mock_db_session, mock_alpha_vantage_service, mock_finnhub_service):
    """Test triggering an update for all supported sectors."""
    # Arrange
    target_date = date.today()
    num_sectors = len(SECTOR_ETF_MAP)
    # Simulate some existing, some new
    mock_db_session.query.return_value.filter.return_value.first.side_effect = \
        [None] * (num_sectors // 2) + [MagicMock(spec=SectorFundamental)] * (num_sectors - num_sectors // 2)

    # Act
    results = await sector_service.update_all_sector_fundamentals(target_date)

    # Assert
    assert isinstance(results, list)
    assert len(results) == num_sectors
    assert all(isinstance(r, SectorFundamentalResponse) for r in results)

    # Verify API calls were made for each sector
    expected_api_calls = num_sectors
    actual_av_calls = mock_alpha_vantage_service.get_overview.call_count
    actual_fh_calls = mock_finnhub_service.get_company_basic_financials.call_count
    assert actual_av_calls + actual_fh_calls == expected_api_calls

    # Verify DB interactions (adds for new, updates handled by commit, refresh for all)
    assert mock_db_session.add.call_count == (num_sectors // 2)
    assert mock_db_session.commit.call_count == num_sectors # Commit happens per sector in the loop
    assert mock_db_session.refresh.call_count == num_sectors

@pytest.mark.asyncio
async def test_fetch_and_store_fundamentals_api_error(sector_service, mock_db_session, mock_alpha_vantage_service):
    """Test handling when an API call fails during fetch."""
    # Arrange
    sector_name = "Financials"
    target_date = date.today()
    etf_symbol = SECTOR_ETF_MAP[sector_name]
    # Simulate AlphaVantage raising an error
    mock_alpha_vantage_service.get_overview.side_effect = HTTPException(status_code=503, detail="AV API unavailable")
    # Assume Finnhub is not called or also fails
    sector_service.fh_service.get_company_basic_financials = AsyncMock(side_effect=Exception("FH Error"))

    # Act & Assert
    with pytest.raises(HTTPException) as exc_info:
        await sector_service._fetch_and_store_fundamentals(sector_name, etf_symbol, target_date)

    assert exc_info.value.status_code == 503 # Should propagate the first error encountered
    assert "AV API unavailable" in exc_info.value.detail

    # Verify no data was committed
    mock_db_session.add.assert_not_called()
    mock_db_session.commit.assert_not_called()
    mock_db_session.rollback.assert_called_once() # Ensure rollback on error

# Add more tests:
# - Test parsing logic for different API response formats (AV vs FH)
# - Test handling of missing data points (e.g., P/E is null)
# - Test specific error scenarios (e.g., API key invalid)
