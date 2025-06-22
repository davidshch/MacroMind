#!/usr/bin/env python3
"""
Script to populate the database with historical price data for volatility model training.
This creates synthetic but realistic price data for the past few years.
"""

import asyncio
import sys
import os
from datetime import datetime, date, timedelta
import random
import numpy as np
import pandas as pd

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# For now, we'll create a simple price data structure that the volatility service can use
# In a real implementation, you'd store this in a database table

def generate_historical_price_data(symbol: str, start_date: date, end_date: date) -> list:
    """Generate realistic historical price data for training."""
    
    # Base prices for different symbols
    base_prices = {
        "NVDA": 150.0,
        "AAPL": 180.0,
        "TSLA": 250.0,
        "MSFT": 400.0,
        "GOOGL": 140.0,
        "AMZN": 180.0,
        "META": 500.0,
        "SPY": 450.0,
        "QQQ": 380.0,
        "^VIX": 20.0
    }
    
    base_price = base_prices.get(symbol, 100.0)
    
    # Generate trading days (skip weekends)
    trading_days = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:  # Monday = 0, Friday = 4
            trading_days.append(current_date)
        current_date += timedelta(days=1)
    
    price_data = []
    current_price = base_price
    
    for i, day in enumerate(trading_days):
        # Generate realistic price movements
        # Add some trend and volatility
        trend_factor = 0.0001 * (i % 252)  # Annual trend
        volatility = 0.02  # 2% daily volatility
        random_factor = np.random.normal(0, volatility)
        
        # Calculate price change
        price_change = trend_factor + random_factor
        current_price *= (1 + price_change)
        
        # Generate OHLC data
        daily_volatility = 0.01  # 1% intraday volatility
        open_price = current_price * (1 + np.random.normal(0, daily_volatility * 0.5))
        high_price = max(open_price, current_price) * (1 + abs(np.random.normal(0, daily_volatility * 0.3)))
        low_price = min(open_price, current_price) * (1 - abs(np.random.normal(0, daily_volatility * 0.3)))
        close_price = current_price
        
        # Generate volume
        base_volume = 1000000
        volume = base_volume * (1 + np.random.normal(0, 0.3))
        
        price_data.append({
            "date": day.strftime("%Y-%m-%d"),
            "open": round(open_price, 2),
            "high": round(high_price, 2),
            "low": round(low_price, 2),
            "close": round(close_price, 2),
            "volume": int(volume)
        })
    
    return price_data

async def create_mock_price_service():
    """Create a mock price service that returns our generated data."""
    
    # Generate data for the symbols we need
    symbols = ["NVDA", "AAPL", "^VIX"]
    start_date = date(2022, 7, 1)
    end_date = date(2024, 7, 15)
    
    price_data_cache = {}
    
    for symbol in symbols:
        print(f"Generating price data for {symbol}...")
        price_data_cache[symbol] = generate_historical_price_data(symbol, start_date, end_date)
        print(f"Generated {len(price_data_cache[symbol])} price records for {symbol}")
    
    # Create a mock market data service
    class MockMarketDataService:
        def __init__(self, price_cache):
            self.price_cache = price_cache
        
        async def get_historical_prices(self, symbol: str, lookback_days: int = 30):
            if symbol in self.price_cache:
                # Return the requested number of days, but from the end of our data
                data = self.price_cache[symbol]
                return data[-lookback_days:] if lookback_days <= len(data) else data
            return []
    
    return MockMarketDataService(price_data_cache)

if __name__ == "__main__":
    # Test the price data generation
    print("Testing price data generation...")
    
    # Generate sample data
    nvda_data = generate_historical_price_data("NVDA", date(2023, 1, 1), date(2023, 1, 31))
    print(f"Generated {len(nvda_data)} NVDA price records")
    print(f"Sample data: {nvda_data[:3]}")
    
    # Test the mock service
    async def test_mock_service():
        mock_service = await create_mock_price_service()
        
        # Test getting historical data
        nvda_prices = await mock_service.get_historical_prices("NVDA", 30)
        print(f"Mock service returned {len(nvda_prices)} NVDA price records")
        print(f"Date range: {nvda_prices[0]['date']} to {nvda_prices[-1]['date']}")
    
    asyncio.run(test_mock_service()) 