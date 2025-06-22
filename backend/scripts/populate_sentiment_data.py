#!/usr/bin/env python3
"""
Script to populate the database with historical sentiment data for volatility model training.
This creates synthetic but realistic sentiment data for the past year.
"""

import asyncio
import sys
import os
from datetime import datetime, date, timedelta
import random
import numpy as np

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.database import get_db
from src.database.models import AggregatedSentiment, SentimentType
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text, select, func

async def populate_historical_sentiment_data():
    """Populate database with historical sentiment data for multiple stocks."""
    
    # Extended list of popular stocks
    symbols = ["NVDA", "AAPL", "TSLA", "MSFT", "GOOGL", "AMZN", "META", "SPY", "QQQ"]
    
    # Extend the date range to cover the full training period plus lookback and recent data for predictions
    start_date = date(2022, 7, 1)  # Earlier start to cover the 90-day lookback
    end_date = date(2025, 6, 30)   # Extended end date to include recent data for predictions
    
    # Get database session
    async for db in get_db():
        try:
            print(f"Populating historical sentiment data for {symbols} from {start_date} to {end_date}")
            
            for symbol in symbols:
                print(f"Processing {symbol}...")
                
                # Check if we already have data for this symbol
                existing_result = await db.execute(
                    select(func.count(AggregatedSentiment.id)).where(AggregatedSentiment.symbol == symbol)
                )
                existing_count = existing_result.scalar()
                
                if existing_count > 100:  # Skip if we already have substantial data
                    print(f"Skipping {symbol} - already has {existing_count} records")
                    continue
                
                # Generate realistic sentiment data
                current_date = start_date
                sentiment_records = []
                
                while current_date <= end_date:
                    # Skip weekends (simple approach)
                    if current_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                        current_date += timedelta(days=1)
                        continue
                    
                    # Generate realistic sentiment scores with some trends and volatility
                    base_score = 0.1  # Slightly bullish baseline
                    
                    # Add some trend over time based on stock performance
                    if symbol == "NVDA":
                        trend_factor = (current_date - start_date).days / 365 * 0.4  # Strong increase for NVDA
                    elif symbol == "TSLA":
                        trend_factor = (current_date - start_date).days / 365 * 0.2  # Moderate increase for TSLA
                    elif symbol == "AAPL":
                        trend_factor = (current_date - start_date).days / 365 * 0.15  # Steady increase for AAPL
                    elif symbol == "MSFT":
                        trend_factor = (current_date - start_date).days / 365 * 0.25  # Good increase for MSFT
                    elif symbol == "GOOGL":
                        trend_factor = (current_date - start_date).days / 365 * 0.1  # Moderate increase for GOOGL
                    elif symbol == "AMZN":
                        trend_factor = (current_date - start_date).days / 365 * 0.05  # Small increase for AMZN
                    elif symbol == "META":
                        trend_factor = (current_date - start_date).days / 365 * 0.3  # Strong increase for META
                    elif symbol in ["SPY", "QQQ"]:
                        trend_factor = (current_date - start_date).days / 365 * 0.12  # Market-like increase
                    else:
                        trend_factor = (current_date - start_date).days / 365 * 0.1  # Default increase
                    
                    # Add some volatility/randomness
                    volatility = 0.2
                    random_factor = np.random.normal(0, volatility)
                    
                    # Calculate final score
                    score = base_score + trend_factor + random_factor
                    score = max(-1.0, min(1.0, score))  # Clamp between -1 and 1
                    
                    # Determine sentiment type
                    if score > 0.15:
                        sentiment = SentimentType.BULLISH
                    elif score < -0.15:
                        sentiment = SentimentType.BEARISH
                    else:
                        sentiment = SentimentType.NEUTRAL
                    
                    # Generate related scores
                    news_score = score + np.random.normal(0, 0.1)
                    reddit_score = score + np.random.normal(0, 0.15)
                    
                    # Create sentiment record
                    sentiment_record = AggregatedSentiment(
                        symbol=symbol,
                        date=current_date,
                        sentiment=sentiment,
                        score=round(score, 3),
                        avg_daily_score=round(score, 3),
                        moving_avg_7d=round(score, 3),  # Simplified
                        news_score=round(news_score, 3),
                        reddit_score=round(reddit_score, 3),
                        benchmark="SPY",
                        timestamp=datetime.now()
                    )
                    
                    sentiment_records.append(sentiment_record)
                    current_date += timedelta(days=1)
                
                # Batch insert all records
                db.add_all(sentiment_records)
                await db.commit()
                
                print(f"Added {len(sentiment_records)} sentiment records for {symbol}")
            
            print("Historical sentiment data population completed successfully!")
            
        except Exception as e:
            print(f"Error populating sentiment data: {e}")
            await db.rollback()
            raise
        finally:
            await db.close()
        break

if __name__ == "__main__":
    asyncio.run(populate_historical_sentiment_data()) 