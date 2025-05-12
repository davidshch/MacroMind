from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
import logging
from typing import AsyncGenerator
from .models import Base

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SQLALCHEMY_DATABASE_URL = os.getenv("DATABASE_URL")
if not SQLALCHEMY_DATABASE_URL:
    logger.error("DATABASE_URL environment variable is not set")
    raise ValueError("DATABASE_URL environment variable is not set")

# Ensure the URL is compatible with asyncpg (e.g., postgresql+asyncpg://...)
if not SQLALCHEMY_DATABASE_URL.startswith("postgresql+asyncpg://"):
    logger.warning(f"DATABASE_URL does not start with postgresql+asyncpg://. Current URL: {SQLALCHEMY_DATABASE_URL}. Adjust if using asyncpg.")
    # Forcing it for asyncpg, ensure your .env is correct
    if SQLALCHEMY_DATABASE_URL.startswith("postgresql://"):
        SQLALCHEMY_DATABASE_URL = SQLALCHEMY_DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)
        logger.info(f"Adjusted DATABASE_URL for asyncpg: {SQLALCHEMY_DATABASE_URL}")

try:
    engine = create_async_engine(SQLALCHEMY_DATABASE_URL, echo=False)
    AsyncSessionLocal = sessionmaker(
        bind=engine, 
        class_=AsyncSession, 
        expire_on_commit=False, 
        autocommit=False, 
        autoflush=False
    )
    logger.info("Async database engine and session maker initialized.")
except Exception as e:
    logger.error(f"Failed to initialize async database: {str(e)}", exc_info=True)
    raise

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        try:
            yield session
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error during yield: {e}", exc_info=True)
            raise
