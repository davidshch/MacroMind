from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey, Enum, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
import enum

Base = declarative_base()

class SentimentType(enum.Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"

class User(Base):
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_vip = Column(Boolean, default=False)
    created_at = Column(DateTime)

class EconomicEvent(Base):
    __tablename__ = "economic_events"
    
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    date = Column(DateTime)
    description = Column(String)
    impact = Column(String)
    forecast = Column(Float, nullable=True)
    previous = Column(Float, nullable=True)
    actual = Column(Float, nullable=True)

class MarketSentiment(Base):
    __tablename__ = "market_sentiments"
    
    id = Column(Integer, primary_key=True, index=True)
    symbol = Column(String, index=True)
    sentiment = Column(Enum(SentimentType))
    score = Column(Float)
    timestamp = Column(DateTime)
