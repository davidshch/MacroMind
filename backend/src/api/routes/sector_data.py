"""
API routes for sector fundamental data.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import List, Optional
from datetime import date
import logging

from ...database.database import get_db
from ...schemas.sector import SectorFundamentalResponse
from ...services.sector_fundamentals import SectorFundamentalsService, SECTOR_ETF_MAP

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/sectors",
    tags=["Sector Data"],
    responses={404: {"description": "Not found"}},
)

def get_sector_fundamentals_service(db: Session = Depends(get_db)) -> SectorFundamentalsService:
    """Dependency to get SectorFundamentalsService."""
    return SectorFundamentalsService(db=db)

@router.get(
    "/{sector_name}/fundamentals",
    response_model=SectorFundamentalResponse,
    summary="Get Fundamental Data for a Specific Sector",
    description="Retrieves fundamental data (P/E, P/B ratios) for a given market sector, proxied by its representative ETF. Data is fetched if not recently cached or stored."
)
async def get_sector_fundamental_data(
    sector_name: str,
    target_date: Optional[date] = Query(None, description="Target date for fundamentals (YYYY-MM-DD). Defaults to today if not provided."),
    service: SectorFundamentalsService = Depends(get_sector_fundamentals_service),
):
    """
    Endpoint to fetch fundamental data for a specific sector.
    - **sector_name**: The name of the sector (e.g., 'technology', 'healthcare').
    - **target_date**: Optional date for which to fetch data. Defaults to the current date.
    """
    try:
        effective_date = target_date if target_date else date.today()
        logger.info(f"Fetching fundamentals for sector: {sector_name} on date: {effective_date}")
        fundamentals = await service.get_sector_fundamentals(sector_name=sector_name.lower(), target_date=effective_date)
        if not fundamentals:
            raise HTTPException(status_code=404, detail=f"Fundamental data not found for sector '{sector_name}' on {effective_date}.")
        return fundamentals
    except HTTPException as http_exc:
        logger.error(f"HTTPException in get_sector_fundamental_data for {sector_name}: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.exception(f"Error fetching fundamentals for sector {sector_name}: {e}")
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred while fetching data for sector '{sector_name}'.")

@router.get(
    "/fundamentals/all",
    response_model=List[SectorFundamentalResponse],
    summary="Update and Get Fundamentals for All Supported Sectors",
    description="Triggers an update (fetch from API if needed) and returns fundamental data for all supported market sectors for a given date. Defaults to today."
)
async def update_and_get_all_sector_fundamentals(
    target_date: Optional[date] = Query(None, description="Target date for fundamentals (YYYY-MM-DD). Defaults to today if not provided."),
    service: SectorFundamentalsService = Depends(get_sector_fundamentals_service),
):
    """
    Endpoint to update and fetch fundamental data for all supported sectors.
    - **target_date**: Optional date for which to fetch/update data. Defaults to the current date.
    """
    try:
        effective_date = target_date if target_date else date.today()
        logger.info(f"Updating and fetching all sector fundamentals for date: {effective_date}")
        updated_fundamentals = await service.update_all_sector_fundamentals(target_date=effective_date)
        if not updated_fundamentals:
            logger.warning(f"No fundamental data could be updated or retrieved for any sector on {effective_date}.")
            # Return empty list if no data, not necessarily an error if APIs are down or no data for that day
            return []
        return updated_fundamentals
    except Exception as e:
        logger.exception(f"Error updating all sector fundamentals: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred while updating sector fundamentals.")

@router.get(
    "/supported-sectors",
    response_model=List[str],
    summary="Get List of Supported Sectors",
    description="Returns a list of all market sectors for which fundamental data can be retrieved."
)
async def get_supported_sectors():
    """
    Endpoint to get a list of supported sectors.
    """
    return list(SECTOR_ETF_MAP.keys())

