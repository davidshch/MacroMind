import asyncio
from src.database.database import get_db
from src.database.models import AggregatedSentiment
from sqlalchemy import select
from datetime import date, timedelta

async def check_tsla_sentiment():
    async for db in get_db():
        try:
            # Check recent sentiment data for TSLA
            result = await db.execute(
                select(AggregatedSentiment)
                .where(AggregatedSentiment.symbol == 'TSLA')
                .order_by(AggregatedSentiment.date.desc())
                .limit(10)
            )
            sentiments = result.scalars().all()
            
            print(f'TSLA sentiments found: {len(sentiments)}')
            for s in sentiments:
                print(f'{s.date}: {s.score}')
                
            # Check if we have any sentiment data in the last 30 days
            thirty_days_ago = date.today() - timedelta(days=30)
            recent_result = await db.execute(
                select(AggregatedSentiment)
                .where(
                    AggregatedSentiment.symbol == 'TSLA',
                    AggregatedSentiment.date >= thirty_days_ago
                )
                .order_by(AggregatedSentiment.date.desc())
            )
            recent_sentiments = recent_result.scalars().all()
            print(f'TSLA sentiments in last 30 days: {len(recent_sentiments)}')
            
        except Exception as e:
            print(f"Error: {e}")
        break

if __name__ == "__main__":
    asyncio.run(check_tsla_sentiment()) 