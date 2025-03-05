from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://username:password@localhost/macromind_db"

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class EconomicEvent(Base):
    __tablename__ = "economic_events"
    id = Column(Integer, primary_key=True, index=True)
    event_name = Column(String, index=True)
    date = Column(String)
    impact = Column(String)

Base.metadata.create_all(bind=engine)