#!/usr/bin/env python3
"""
Script to check sentiment data in the database.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.database import get_db
from src.database.models import AggregatedSentiment, MarketSentiment
from sqlalchemy import select

async def check_sentiment_data():
    """Check what sentiment data exists in the database."""
    
    async for db in get_db():
        try:
            print("Checking sentiment data in database...")
            
            # Check AggregatedSentiment table
            result = await db.execute(select(AggregatedSentiment).where(AggregatedSentiment.symbol == 'NVDA'))
            aggregated_records = result.scalars().all()
            print(f"Found {len(aggregated_records)} NVDA records in AggregatedSentiment table")
            
            if aggregated_records:
                print(f"Sample dates: {[r.date for r in aggregated_records[:5]]}")
                print(f"Date range: {min(r.date for r in aggregated_records)} to {max(r.date for r in aggregated_records)}")
            
            # Check MarketSentiment table
            result = await db.execute(select(MarketSentiment).where(MarketSentiment.symbol == 'NVDA'))
            market_records = result.scalars().all()
            print(f"Found {len(market_records)} NVDA records in MarketSentiment table")
            
            if market_records:
                print(f"Sample dates: {[r.date for r in market_records[:5]]}")
                print(f"Date range: {min(r.date for r in market_records)} to {max(r.date for r in market_records)}")
            
            # Check what table the volatility service is actually querying
            print("\nChecking what table the volatility service queries...")
            
        except Exception as e:
            print(f"Error checking sentiment data: {e}")
            raise
        finally:
            await db.close()
        break

if __name__ == "__main__":
    asyncio.run(check_sentiment_data()) 