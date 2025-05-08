"""Economic calendar service for tracking and analyzing market events."""

from typing import List, Dict, Any, Optional
import aiohttp
from datetime import datetime, timedelta, date
from sqlalchemy.orm import Session
from ..config import get_settings
from ..database.models import EconomicEvent
# Import EventResponse schema
from ..schemas.event import EventResponse
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

class EconomicCalendarService:
    """Manages economic events and impact analysis."""
    
    def __init__(self, db: Optional[Session] = None):
        self.db = db
        self.cache = {}
        self.cache_duration = timedelta(hours=4)

    async def get_economic_events(
        self, 
        start_date: date = None, 
        end_date: date = None,
        importance: str = "high"
    ) -> List[EventResponse]: # Changed return type hint
        """Get economic events within date range."""
        if not start_date:
            start_date = date.today()
        if not end_date:
            end_date = start_date + timedelta(days=7)

        cache_key = f"events_{start_date}_{end_date}_{importance}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            # Ensure cached data is the correct type (or re-validate)
            if isinstance(cached_data, list) and (not cached_data or isinstance(cached_data[0], EventResponse)):
                 return cached_data
            else:
                 # Invalidate cache if type is wrong
                 logger.warning(f"Invalidating cache for key {cache_key} due to type mismatch.")
                 self._remove_from_cache(cache_key)

        try:
            # Fetch DB models
            db_events = await self._fetch_economic_events_from_db(start_date, end_date, importance)
            
            # Convert DB models to Pydantic EventResponse models
            # Use model_validate for Pydantic v2
            events_response = [EventResponse.model_validate(event) for event in db_events]
            
            self._add_to_cache(cache_key, events_response) # Cache the Pydantic models
            return events_response
        except Exception as e:
            logger.error(f"Error fetching economic events: {str(e)}")
            # Return empty list on error to avoid breaking clients, but log the error
            return []

    async def _fetch_economic_events_from_db(
        self,
        start_date: date,
        end_date: date,
        importance: str = "high"
    ) -> List[EconomicEvent]: # Return DB models
        """Fetch economic events from the database.

        Args:
            start_date: The start date for the query.
            end_date: The end date for the query.
            importance: The minimum importance level (e.g., 'high', 'medium', 'low', 'all').

        Returns:
            A list of EconomicEvent database models.
        """
        if not self.db:
            logger.error("Database session not available in EconomicCalendarService")
            return []
        try:
            # Query database for existing events
            query = self.db.query(EconomicEvent).filter(
                EconomicEvent.date >= start_date,
                # Add 1 day to end_date to make it inclusive for date comparisons
                EconomicEvent.date < (end_date + timedelta(days=1)) 
            )
            
            # Filter by importance, handling 'all' case and case-insensitivity
            importance_upper = importance.upper() if importance else None
            if importance_upper and importance_upper != 'ALL':
                 # Assuming impact is stored as uppercase in the DB ('HIGH', 'MEDIUM', 'LOW')
                query = query.filter(EconomicEvent.impact == importance_upper)

            events = query.order_by(EconomicEvent.date).all()
            return events

        except Exception as e:
            logger.error(f"Error in _fetch_economic_events_from_db: {str(e)}")
            # Re-raise or return empty list based on desired error handling
            return []

    # Removed the old _fetch_economic_events method as logic is merged/split

    async def add_economic_event(
        self,
        name: str,
        date: datetime,
        impact: str,
        description: str,
        forecast: Optional[float] = None,
        previous: Optional[float] = None,
        actual: Optional[float] = None
    ) -> EconomicEvent:
        """Add a new economic event to the database."""
        if not self.db:
            logger.error("Database session not available for adding event")
            raise Exception("Database session not configured") # Or a more specific exception
        try:
            event = EconomicEvent(
                name=name,
                date=date,
                # Ensure impact is stored consistently (e.g., uppercase)
                impact=impact.upper(), 
                description=description,
                forecast=forecast,
                previous=previous,
                actual=actual
            )
            self.db.add(event)
            self.db.commit()
            self.db.refresh(event)
            logger.info(f"Added economic event: {name} ({date})")
            return event
        except Exception as e:
            logger.error(f"Error adding economic event: {str(e)}")
            self.db.rollback()
            raise

    def _get_from_cache(self, key: str) -> Any:
        """Get cached data if still valid."""
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.cache_duration:
                logger.debug(f"Cache hit for key: {key}")
                return data
            logger.debug(f"Cache expired for key: {key}")
            del self.cache[key]
        logger.debug(f"Cache miss for key: {key}")
        return None

    def _add_to_cache(self, key: str, data: Any):
        """Cache data with timestamp."""
        self.cache[key] = (data, datetime.now())
        logger.debug(f"Cached data for key: {key}")

    def _remove_from_cache(self, key: str):
        """Remove data from cache."""
        if key in self.cache:
            del self.cache[key]
            logger.debug(f"Removed data from cache for key: {key}")
