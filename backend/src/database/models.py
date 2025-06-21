"""
Database models for MacroMind platform.

This module contains SQLAlchemy ORM models that define the database schema
for the MacroMind financial analytics platform. Each model corresponds to
a table in the PostgreSQL database.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum, Boolean, JSON, Index, Date
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum
from datetime import datetime

Base = declarative_base()

class SentimentType(enum.Enum):
    """Market sentiment classification."""
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class User(Base):
    """
    User model with authentication and preferences.
    
    Attributes:
        email: Unique email for login
        hashed_password: Securely stored password
        is_vip: Premium status
        alerts: User's configured alerts
    """
    
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_vip = Column(Boolean, default=False)
    created_at = Column(DateTime)
    alerts = relationship("Alert", back_populates="user")

class EconomicEvent(Base):
    """
    Economic calendar event tracking.

    Attributes:
        name: Event name (e.g., "FOMC Meeting")
        date: Scheduled event time
        impact: Expected market impact
        forecast/previous/actual: Event values
    """
    
    __tablename__ = "economic_events"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    date = Column(DateTime)
    description = Column(String)
    impact = Column(String)
    forecast = Column(Float, nullable=True)
    previous = Column(Float, nullable=True)
    actual = Column(Float, nullable=True)
    
    __table_args__ = (
        Index('idx_date_impact', 'date', 'impact'),
    )

class MarketSentiment(Base):
    """
    Market sentiment snapshot, including aggregated and source-specific scores.

    Attributes:
        id: Primary key.
        symbol: Asset symbol (e.g., "AAPL", "BTC").
        date: Date of the sentiment record.
        sentiment: Overall aggregated sentiment (bullish/bearish/neutral).
        score: Overall aggregated confidence score (normalized -1 to 1).
        avg_daily_score: Average combined score for the day.
        moving_avg_7d: 7-day moving average of the combined score.
        news_score: Normalized sentiment score from news sources.
        reddit_score: Normalized sentiment score from Reddit sources.
        benchmark: Benchmark used for comparison (SPY, QQQ, RSP).
        market_condition: Market condition (e.g., volatile, normal).
        volatility_level: Current volatility level.
        timestamp: Timestamp of the last update.
    """

    __tablename__ = "market_sentiments"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    date = Column(Date, index=True)  # Added date field
    sentiment = Column(Enum(SentimentType))
    score = Column(Float) # Normalized score (-1 to 1)
    avg_daily_score = Column(Float, nullable=True) # Combined daily average
    moving_avg_7d = Column(Float, nullable=True) # 7-day moving average
    news_score = Column(Float, nullable=True) # Normalized news score
    reddit_score = Column(Float, nullable=True) # Normalized reddit score
    benchmark = Column(String, nullable=True) # SPY, QQQ, RSP
    market_condition = Column(String, nullable=True) # Added: e.g., volatile, normal
    volatility_level = Column(Float, nullable=True) # Added: Current volatility value
    timestamp = Column(DateTime)

    __table_args__ = (
        Index('idx_symbol_date', 'symbol', 'date'), # Updated index
    )

class SectorFundamental(Base):
    """
    Stores basic fundamental data for major market sectors.

    Attributes:
        id: Primary key.
        sector_name: Name of the sector (e.g., Technology, Healthcare).
        date: Date of the fundamental data record.
        pe_ratio: Price-to-Earnings ratio.
        pb_ratio: Price-to-Book ratio.
        earnings_growth: Earnings growth rate (%).
        timestamp: Timestamp of the last update.
    """
    __tablename__ = "sector_fundamentals"

    id = Column(Integer, primary_key=True, index=True)
    sector_name = Column(String, index=True)
    date = Column(Date, index=True)
    pe_ratio = Column(Float, nullable=True)
    pb_ratio = Column(Float, nullable=True)
    earnings_growth = Column(Float, nullable=True)
    timestamp = Column(DateTime)

    __table_args__ = (
        Index('idx_sector_date', 'sector_name', 'date'),
    )

class Alert(Base):
    """User-configured market alerts."""
    
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    name = Column(String, nullable=True) # Added name field
    symbol = Column(String)
    conditions = Column(JSON) # Stores the AlertConditions schema
    created_at = Column(DateTime, default=datetime.utcnow) # Changed to utcnow for consistency
    is_active = Column(Boolean, default=True)
    last_triggered_at = Column(DateTime, nullable=True) # Renamed from last_triggered

    user = relationship("User", back_populates="alerts")

class RawSentimentAnalysis(Base):
    """
    Stores individual sentiment analysis results from FinBERT for raw texts.
    Used by Insight Distiller and potentially for fine-tuning.
    """
    __tablename__ = "raw_sentiment_analyses"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True, nullable=True) # Symbol might not always be directly applicable to any text
    text_content_hash = Column(String, index=True, unique=True) # To avoid duplicate processing of same text
    text_content = Column(String)
    source = Column(String, index=True) # e.g., "NewsAPI", "Reddit_PostTitle", "Reddit_Comment"
    sentiment_label = Column(String) # e.g., "positive", "negative", "neutral" from FinBERT
    sentiment_score = Column(Float)  # Confidence for the primary label
    all_scores = Column(JSON, nullable=True) # Stores all scores from FinBERT if available
    analyzed_at = Column(DateTime, default=datetime.utcnow, index=True)
    source_created_at = Column(DateTime, nullable=True, index=True) # Original timestamp of the news/post
    # Optional: link to a parent economic event or a specific news article ID
    # economic_event_id = Column(Integer, ForeignKey("economic_events.id"), nullable=True)
    # news_article_id = Column(String, nullable=True, index=True) # If we have unique IDs for news

    __table_args__ = (
        Index('idx_raw_sentiment_symbol_source_created', 'symbol', 'source', 'source_created_at'),
    )

class AggregatedSentiment(Base):
    """Aggregated sentiment data for a specific symbol and date."""
    __tablename__ = "aggregated_sentiments"

    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    date = Column(Date, index=True)
    sentiment = Column(Enum(SentimentType))
    score = Column(Float)
    avg_daily_score = Column(Float, nullable=True)
    moving_avg_7d = Column(Float, nullable=True)
    news_score = Column(Float, nullable=True)
    reddit_score = Column(Float, nullable=True)
    benchmark = Column(String, nullable=True)
    timestamp = Column(DateTime)
