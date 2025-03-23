"""
Database models for MacroMind platform.

This module contains SQLAlchemy ORM models that define the database schema
for the MacroMind financial analytics platform. Each model corresponds to
a table in the PostgreSQL database.
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum, Boolean, JSON, Index
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
    Market sentiment snapshot.

    Attributes:
        symbol: Asset symbol
        sentiment: Bullish/bearish/neutral
        score: Confidence score
    """
    
    __tablename__ = "market_sentiments"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    sentiment = Column(Enum(SentimentType))
    score = Column(Float)
    timestamp = Column(DateTime)
    
    __table_args__ = (
        Index('idx_symbol_timestamp', 'symbol', 'timestamp'),
    )

class Alert(Base):
    """User-configured market alerts."""
    
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    symbol = Column(String)
    alert_type = Column(String)  # price, volatility, sentiment
    condition = Column(JSON)
    created_at = Column(DateTime, default=datetime.now())
    is_active = Column(Boolean, default=True)
    last_checked = Column(DateTime)
    last_triggered = Column(DateTime, nullable=True)

    user = relationship("User", back_populates="alerts")
