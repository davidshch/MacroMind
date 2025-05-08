"""Sector fundamentals service for tracking sector ETFs and metrics."""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta, date
import asyncio
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import and_
from fastapi import HTTPException
from ..database.models import SectorFundamental
from ..schemas.sector import SectorFundamentalResponse, SectorFundamentalCreate
from ..config import get_settings
from .market_data_services import AlphaVantageService, FinnhubService
import os

logger = logging.getLogger(__name__)
settings = get_settings()

# Map sectors to their corresponding ETFs
SECTOR_ETF_MAP = {
    "technology": "XLK",
    "healthcare": "XLV",
    "financials": "XLF",
    "consumer_discretionary": "XLY",
    "consumer_staples": "XLP",
    "energy": "XLE",
    "materials": "XLB",
    "industrials": "XLI",
    "utilities": "XLU",
    "real_estate": "XLRE",
    "communication_services": "XLC"
}

class SectorFundamentalsService:
    """Service for managing sector fundamental data using ETFs as proxies."""
    
    def __init__(
        self, 
        db: Session, 
        alpha_vantage_service: Optional[AlphaVantageService] = None,
        finnhub_service: Optional[FinnhubService] = None
    ):
        """Initialize the sector fundamentals service.
        
        Args:
            db: Database session
            alpha_vantage_service: Optional AlphaVantageService instance
            finnhub_service: Optional FinnhubService instance
        """
        self.db = db
        self.alpha_vantage = alpha_vantage_service or AlphaVantageService()
        self.finnhub = finnhub_service or FinnhubService()
        self.cache = {}
        self.cache_duration = timedelta(hours=24)
    
    async def get_sector_fundamentals(
        self, 
        sector_name: str, 
        target_date: date = date.today()
    ) -> SectorFundamentalResponse:
        """Get fundamental data for a specific sector.
        
        Args:
            sector_name: Sector name to get data for (must be in SECTOR_ETF_MAP)
            target_date: Date to get fundamentals for (defaults to today)
            
        Returns:
            SectorFundamentalResponse containing sector fundamentals
            
        Raises:
            HTTPException: If sector is invalid or data cannot be retrieved
        """
        # Validate sector name
        sector_name = sector_name.lower()
        if sector_name not in SECTOR_ETF_MAP:
            logger.error(f"Invalid sector requested: {sector_name}")
            raise HTTPException(
                status_code=404, 
                detail=f"Sector '{sector_name}' not supported or recognized."
            )
            
        # Check DB first
        db_data = self.db.query(SectorFundamental).filter(
            SectorFundamental.sector_name == sector_name,
            SectorFundamental.date == target_date
        ).first()

        if db_data:
            return SectorFundamentalResponse.model_validate(db_data)

        # If not in DB, fetch, store, and return
        fetched_data = await self._fetch_and_store_sector_fundamentals(sector_name, target_date)
        if fetched_data:
            return SectorFundamentalResponse.model_validate(fetched_data)
        else:
            # Could not fetch data
            raise HTTPException(
                status_code=500, 
                detail=f"Could not fetch fundamental data for sector '{sector_name}'."
            )
    
    async def update_all_sector_fundamentals(self, target_date: date = date.today()) -> List[SectorFundamentalResponse]:
        """Update fundamentals for all sectors.
        
        Args:
            target_date: Date to update fundamentals for
            
        Returns:
            List of updated SectorFundamentalResponse objects
        """
        results = []
        for sector_name in SECTOR_ETF_MAP.keys():
            try:
                fundamental = await self._fetch_and_store_sector_fundamentals(sector_name, target_date)
                if fundamental:
                    results.append(SectorFundamentalResponse.model_validate(fundamental))
                # Basic rate limiting for API calls - REMOVED for testing efficiency
                # await asyncio.sleep(1) 
            except Exception as e:
                logger.error(f"Failed to update fundamentals for sector {sector_name}: {str(e)}")
        
        return results
    
    async def _fetch_and_store_sector_fundamentals(
        self, 
        sector_name: str, 
        target_date: date
    ) -> Optional[SectorFundamental]:
        """Fetches data from external APIs for a sector ETF and stores it.
        
        Args:
            sector_name: Sector name
            target_date: Date to fetch data for
            
        Returns:
            SectorFundamental object or None if failed
        """
        etf_symbol = SECTOR_ETF_MAP.get(sector_name)
        if not etf_symbol:
            logger.warning(f"No ETF mapping found for sector: {sector_name}")
            return None

        # Check cache first
        cache_key = f"fundamentals_{sector_name}_{target_date}"
        cached_data = self._get_from_cache(cache_key)
        if cached_data:
            # Check if cached data is already stored in DB for the date
            existing_db = self.db.query(SectorFundamental).filter(
                SectorFundamental.sector_name == sector_name,
                SectorFundamental.date == target_date
            ).first()
            if existing_db:
                return existing_db  # Already fetched and stored
                
        try:
            logger.info(f"Fetching Finnhub basic financials for {etf_symbol} (proxy for {sector_name})")
            # Use asyncio.to_thread to run the synchronous Finnhub client call
            metrics_data = await asyncio.to_thread(
                self.finnhub.get_company_metrics,
                etf_symbol
            )

            if not metrics_data or 'metric' not in metrics_data:
                logger.warning(f"No metrics data received from Finnhub for {etf_symbol}")
                return None

            metrics = metrics_data['metric']

            # Extract desired fundamentals
            pe_ratio = metrics.get('peNormalizedAnnual') or metrics.get('peBasicExclExtraTTM')
            pb_ratio = metrics.get('pbAnnual') or metrics.get('pbQuarterly')
            earnings_growth = metrics.get('epsGrowthTTMYoy')

            # Create fundamental data object
            fundamental_data = SectorFundamentalCreate(
                sector_name=sector_name,
                date=target_date,
                pe_ratio=float(pe_ratio) if pe_ratio is not None else None,
                pb_ratio=float(pb_ratio) if pb_ratio is not None else None,
                earnings_growth=float(earnings_growth) if earnings_growth is not None else None,
                timestamp=datetime.now()
            )

            # Add to cache
            self._add_to_cache(cache_key, fundamental_data.model_dump())

            # Upsert into database
            existing_record = self.db.query(SectorFundamental).filter(
                SectorFundamental.sector_name == sector_name,
                SectorFundamental.date == target_date
            ).first()

            if existing_record:
                # Update existing record
                existing_record.pe_ratio = fundamental_data.pe_ratio
                existing_record.pb_ratio = fundamental_data.pb_ratio
                existing_record.earnings_growth = fundamental_data.earnings_growth
                existing_record.timestamp = fundamental_data.timestamp
                self.db.commit()
                self.db.refresh(existing_record)
                return existing_record
            else:
                # Create new record
                new_record = SectorFundamental(**fundamental_data.model_dump())
                self.db.add(new_record)
                self.db.commit()
                self.db.refresh(new_record)
                return new_record

        except Exception as e:
            logger.error(f"Error fetching or storing fundamentals for {sector_name}: {str(e)}")
            return None  # Indicate failure
    
    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached data if still valid.
        
        Args:
            key: Cache key
            
        Returns:
            Cached data or None if not found or expired
        """
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.cache_duration:
                return data
            # Cache expired
            del self.cache[key]
        return None
    
    def _add_to_cache(self, key: str, data: Dict[str, Any]):
        """Cache data with timestamp.
        
        Args:
            key: Cache key
            data: Data to cache
        """
        self.cache[key] = (data, datetime.now())
