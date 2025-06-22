"""Economic calendar service for tracking and analyzing market events."""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta, date
from sqlalchemy.ext.asyncio import AsyncSession  # Changed from sqlalchemy.orm import Session
from fastapi import HTTPException, Depends  # Added Depends
import asyncio

from ..config import get_settings
from ..database.models import EconomicEvent
from ..database.database import get_db  # For AsyncSession dependency
from ..schemas.event import EventResponse, EventEchoResponse, EventEchoPatternPoint, EventCreate  # Added EventCreate
from src.services.ml.model_factory import MLModelFactory, get_ml_model_factory
from .market_data import MarketDataService
from sqlalchemy import select, and_, delete, update  # Keep other sqlalchemy imports as needed by other methods
from sqlalchemy.exc import SQLAlchemyError
import pandas as pd
import logging

logger = logging.getLogger(__name__)
settings = get_settings()

class EconomicCalendarService:
    """Manages economic events and impact analysis."""
    
    def __init__(self, 
                 db: AsyncSession = Depends(get_db),  # Changed to AsyncSession
                 ml_model_factory: MLModelFactory = Depends(get_ml_model_factory),
                 market_data_service: MarketDataService = None
                ):
        self.db = db
        self.ml_model_factory = ml_model_factory
        self.market_data_service = market_data_service or MarketDataService()  # Create if not provided
        self.cache = {}
        self.cache_duration = timedelta(hours=4)

    async def get_economic_events(
        self, 
        start_date: Optional[date] = None, 
        end_date: Optional[date] = None,
        importance: Optional[str] = "high",
        country: Optional[str] = None, 
        event_name_like: Optional[str] = None
    ) -> List[EventResponse]:
        """Get economic events within date range, including AI-generated impact scores."""
        if not start_date:
            start_date = date.today()
        if not end_date:
            end_date = start_date + timedelta(days=7)

        cache_key = f"events_v3_{start_date}_{end_date}_{importance}_{country}_{event_name_like}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            return cached_data

        try:
            db_events_models = await self._fetch_economic_events_from_db(
                start_date, end_date, importance, country, event_name_like
            )
            
            events_response_list = []
            if db_events_models:
                for event_model in db_events_models:
                    event_details_for_ai = {
                        "name": event_model.name,
                        "impact": event_model.impact,
                        "currency": getattr(event_model, 'currency', None), 
                        "country": getattr(event_model, 'country', None)
                    }
                    
                    impact_prediction = await self.ml_model_factory.forecast_event_impact_score(event_details_for_ai)
                    
                    event_resp = EventResponse.model_validate(event_model)
                    event_resp.impact_score = impact_prediction.get('impact_score')
                    event_resp.impact_confidence = impact_prediction.get('confidence')
                    events_response_list.append(event_resp)
            
            self._add_to_cache(cache_key, events_response_list)
            return events_response_list
        except SQLAlchemyError as e:
            logger.error(f"Database error fetching economic events: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Database error while fetching events.")
        except Exception as e:
            logger.exception(f"Error fetching economic events with AI impact score: {str(e)}")
            raise HTTPException(status_code=500, detail="Server error while fetching events.")

    async def _fetch_economic_events_from_db(
        self,
        start_date: date,
        end_date: date,
        importance: Optional[str] = "high",
        country: Optional[str] = None,
        event_name_like: Optional[str] = None
    ) -> List[EconomicEvent]:
        """Fetch economic events from the database using AsyncSession."""
        stmt = select(EconomicEvent).filter(
            EconomicEvent.date >= start_date,
            EconomicEvent.date < (end_date + timedelta(days=1))  # Make end_date inclusive for the day
        )
        
        if importance and importance.lower() != 'all':
            stmt = stmt.filter(EconomicEvent.impact == importance.upper())
        if country:
            # Assuming country codes are stored consistently (e.g., uppercase)
            stmt = stmt.filter(EconomicEvent.country == country.upper())
        if event_name_like:
            stmt = stmt.filter(EconomicEvent.name.ilike(f"%{event_name_like}%"))

        stmt = stmt.order_by(EconomicEvent.date)
        result = await self.db.execute(stmt)
        events = result.scalars().all()
        return list(events)

    async def add_economic_event(self, event_data: EventCreate) -> EventResponse:
        """Add a new economic event to the database using AsyncSession."""
        db_event = EconomicEvent(**event_data.model_dump())
        if hasattr(db_event, 'impact') and db_event.impact:
            db_event.impact = db_event.impact.upper()
        
        self.db.add(db_event)
        try:
            await self.db.commit()
            await self.db.refresh(db_event)
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"Database error adding economic event '{event_data.name}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Could not save economic event due to database error.")
        except Exception as e: # Catch any other unexpected errors during commit/refresh
            await self.db.rollback()
            logger.error(f"Unexpected error adding economic event '{event_data.name}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Server error while adding event.")
        
        logger.info(f"Added economic event: {db_event.name} ({db_event.date})")
        
        event_details_for_ai = {
            "name": db_event.name,
            "impact": db_event.impact,
            "currency": getattr(db_event, 'currency', None),
            "country": getattr(db_event, 'country', None)
        }
        impact_prediction = await self.ml_model_factory.forecast_event_impact_score(event_details_for_ai)
        
        response = EventResponse.model_validate(db_event)
        response.impact_score = impact_prediction.get('impact_score')
        response.impact_confidence = impact_prediction.get('confidence')
        return response

    async def get_event_by_id(self, event_id: int) -> Optional[EconomicEvent]:
        """Fetches a single economic event by its ID using AsyncSession."""
        try:
            event = await self.db.get(EconomicEvent, event_id)
            return event
        except SQLAlchemyError as e:
            logger.error(f"Database error fetching event by ID {event_id}: {e}", exc_info=True)
            return None 

    async def get_event_echo_data(
        self,
        reference_event_id: int,
        asset_symbol: str,
        lookback_years: int = 5,
        window_pre_event_days: int = 5,
        window_post_event_days: int = 10,
        min_past_events: int = 3
    ) -> EventEchoResponse:
        """Generates the economic event echo pattern using AsyncSession for DB calls."""
        reference_event_model = await self.get_event_by_id(reference_event_id)

        if not reference_event_model:
            raise HTTPException(status_code=404, detail=f"Reference event with ID {reference_event_id} not found.")

        event_name_to_match = reference_event_model.name
        event_country_to_match = getattr(reference_event_model, 'country', None)
        ref_event_date = reference_event_model.date.date() if isinstance(reference_event_model.date, datetime) else reference_event_model.date

        past_event_date_limit = ref_event_date - timedelta(days=1)
        historical_start_date = past_event_date_limit - timedelta(days=lookback_years * 365)

        stmt = select(EconomicEvent.date).where(
            EconomicEvent.name == event_name_to_match,
            EconomicEvent.date >= historical_start_date,
            EconomicEvent.date <= past_event_date_limit,
            EconomicEvent.id != reference_event_id
        )
        if event_country_to_match: 
             stmt = stmt.where(EconomicEvent.country == event_country_to_match)
        stmt = stmt.order_by(EconomicEvent.date.desc())
        
        try:
            past_event_dates_result = await self.db.execute(stmt)
            past_event_dates_dt = [row[0] for row in past_event_dates_result.fetchall()]
            past_event_dates_for_ml = [dt.date() if isinstance(dt, datetime) else dt for dt in past_event_dates_dt]
        except SQLAlchemyError as e:
            logger.error(f"DB error fetching past similar events for '{event_name_to_match}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error fetching historical event data.")

        if len(past_event_dates_for_ml) < min_past_events:
            msg = f"Not enough similar past events found ({len(past_event_dates_for_ml)}) for '{event_name_to_match}' (Country: {event_country_to_match}) to generate a reliable echo pattern. Minimum required: {min_past_events}."
            logger.warning(msg)
            return EventEchoResponse(
                event_name=reference_event_model.name,
                asset_symbol=asset_symbol,
                reference_event_date=ref_event_date,
                pattern_description="", echo_pattern=[],
                historical_events_analyzed=len(past_event_dates_for_ml), message=msg
            )

        if not past_event_dates_for_ml:
            raise HTTPException(status_code=400, detail="No past event dates available for price data fetching after filtering.")

        min_date_for_prices = min(past_event_dates_for_ml) - timedelta(days=window_pre_event_days + 5)

        required_lookback_days = (date.today() - min_date_for_prices).days + 1
        if required_lookback_days <= 0: 
            required_lookback_days = 365 

        try:
            price_data_list = await self.market_data_service.get_historical_prices(asset_symbol, lookback_days=required_lookback_days)
            if not price_data_list:
                raise ValueError("No historical price data returned from MarketDataService.")

            historical_prices_df = pd.DataFrame(price_data_list)
            historical_prices_df['ds'] = pd.to_datetime(historical_prices_df['date'])
            historical_prices_df = historical_prices_df.rename(columns={"close": "y"})
            historical_prices_df = historical_prices_df[['ds', 'y']].sort_values(by='ds').dropna()
            
            max_date_for_prices_dt = max(past_event_dates_for_ml) + timedelta(days=window_post_event_days + 5)
            historical_prices_df = historical_prices_df[
                (historical_prices_df['ds'] >= pd.to_datetime(min_date_for_prices)) &
                (historical_prices_df['ds'] <= pd.to_datetime(max_date_for_prices_dt))
            ]
            if historical_prices_df.empty:
                 raise ValueError("Historical price data is empty after filtering for the required event windows.")

        except Exception as e:
            logger.error(f"Error fetching/processing historical prices for {asset_symbol} for event echo: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error fetching historical price data for {asset_symbol}.")

        echo_analysis_result = await self.ml_model_factory.analyze_event_echo_with_prophet(
            historical_prices_df=historical_prices_df,
            past_event_dates=past_event_dates_for_ml, 
            window_pre_event_days=window_pre_event_days,
            window_post_event_days=window_post_event_days,
            min_events_for_pattern=min_past_events
        )
        
        return EventEchoResponse(
            event_name=reference_event_model.name,
            asset_symbol=asset_symbol,
            reference_event_date=ref_event_date,
            pattern_description=f"Average price movement pattern around {echo_analysis_result.get('events_analyzed',0)} similar past events for {asset_symbol}. Window: T-{window_pre_event_days} to T+{window_post_event_days} days.",
            echo_pattern=[EventEchoPatternPoint(**point) for point in echo_analysis_result.get("echo_pattern", [])],
            historical_events_analyzed=echo_analysis_result.get("events_analyzed", 0),
            message=echo_analysis_result.get("message", "Analysis complete.")
        )

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

# Dependency provider for EconomicCalendarService
async def get_economic_calendar_service(
    db: AsyncSession = Depends(get_db),
    ml_model_factory: MLModelFactory = Depends(get_ml_model_factory)
) -> EconomicCalendarService:
    return EconomicCalendarService(
        db=db, 
        ml_model_factory=ml_model_factory
    )
