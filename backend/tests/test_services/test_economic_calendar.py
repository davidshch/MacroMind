import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import date, timedelta, datetime
import pandas as pd
from sqlalchemy.orm import Session
from fastapi import HTTPException

from src.services.economic_calendar import EconomicCalendarService
from src.database.models import EconomicEvent
from src.schemas.event import EventResponse

# Mock database session fixture
@pytest.fixture
def mock_db_session():
    session = MagicMock(spec=Session)
    query_mock = MagicMock()
    session.query.return_value = query_mock
    query_mock.filter.return_value = query_mock
    query_mock.order_by.return_value = query_mock
    # Simulate finding some events
    mock_event = EconomicEvent(
        id=1,
        name="Test Event",
        date=datetime.now(),
        impact="High",
        description="A test event",
        forecast=1.0, previous=0.9, actual=None
    )
    query_mock.all.return_value = [mock_event]
    query_mock.first.return_value = mock_event
    return session

# Mock MLModelFactory fixture (if needed for impact prediction)
@pytest.fixture
def mock_ml_factory():
    factory = MagicMock()
    # Mock prediction method if used by the service
    # factory.predict_event_impact = AsyncMock(return_value=...) 
    return factory

@pytest.mark.asyncio
async def test_get_economic_events(mock_db_session):
    """Test retrieving economic events from the database."""
    # Arrange
    service = EconomicCalendarService(db=mock_db_session)
    start_date = date.today()
    end_date = date.today()

    # Act
    events = await service.get_economic_events(start_date, end_date)

    # Assert
    assert isinstance(events, list)
    assert len(events) > 0
    assert isinstance(events[0], EventResponse)
    assert events[0].name == "Test Event"
    mock_db_session.query.assert_called_once_with(EconomicEvent)
    # Add more specific filter/order_by assertions if needed

@pytest.mark.asyncio
async def test_add_economic_event(mock_db_session):
    """Test adding a new economic event."""
    # Arrange
    service = EconomicCalendarService(db=mock_db_session)
    event_data = {
        "name": "New Test Event",
        "date": datetime.now(),
        "impact": "Low",
        "description": "Another test event"
    }

    # Act
    new_event = await service.add_economic_event(**event_data)

    # Assert
    assert isinstance(new_event, EconomicEvent)
    assert new_event.name == "New Test Event"
    mock_db_session.add.assert_called_once()
    mock_db_session.commit.assert_called_once()
    mock_db_session.refresh.assert_called_once()

# Add more tests for:
# - Updating events
# - Predicting event impact (if implemented in service)
# - Filtering by impact, date range, etc.
# - Error handling
