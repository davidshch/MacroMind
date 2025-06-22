#!/usr/bin/env python3
"""
Script to test historical price data fetching.
"""

import asyncio
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.services.market_data import MarketDataService

async def test_historical_prices():
    """Test historical price data fetching."""
    
    market_service = MarketDataService()
    
    print("Testing historical price data for NVDA...")
    
    # Test with different lookback periods
    for days in [30, 90, 365]:
        print(f"\nTesting {days} days lookback...")
        try:
            price_data = await market_service.get_historical_prices('NVDA', days)
            if price_data:
                print(f"Success! Got {len(price_data)} price records")
                print(f"Date range: {price_data[0]['date']} to {price_data[-1]['date']}")
                print(f"Sample record: {price_data[0]}")
            else:
                print("No price data returned")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(test_historical_prices()) 