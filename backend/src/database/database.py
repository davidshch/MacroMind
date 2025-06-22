from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv
import logging
from typing import AsyncGenerator
from .models import Base
from sqlalchemy import text

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
    engine = create_async_engine(
        SQLALCHEMY_DATABASE_URL,
        echo=False,
        pool_size=10,  # Increased pool size
        max_overflow=20,  # Increased max overflow
        pool_timeout=60,  # Increased timeout
        pool_recycle=3600,  # Recycle connections after 1 hour
        pool_pre_ping=True,  # Enable connection health checks
        connect_args={
            "command_timeout": 30,  # Increased timeout for operations
            "server_settings": {
                "statement_timeout": "30000",  # 30 seconds in milliseconds
                "idle_in_transaction_session_timeout": "60000"  # 60 seconds in milliseconds
            }
        },
        # Add retry logic for connection issues
        pool_reset_on_return='commit',  # Reset connection state on return
        pool_use_lifo=True,  # Use LIFO to reduce number of connections
    )
    from sqlalchemy.ext.asyncio import async_sessionmaker

    AsyncSessionLocal = async_sessionmaker(
        bind=engine,
        expire_on_commit=False,
        autoflush=False,
        autocommit=False,
        # Add session configuration for better error handling
        class_=AsyncSession,
    )
    logger.info("Async database engine and session maker initialized.")
except Exception as e:
    logger.error(f"Failed to initialize async database: {str(e)}", exc_info=True)
    raise

async def get_db() -> AsyncGenerator[AsyncSession, None]:
    session = AsyncSessionLocal()
    try:
        # Test the connection before yielding
        await session.execute(text("SELECT 1"))
        yield session
    except Exception as e:
        try:
            await session.rollback()
        except Exception as rollback_error:
            logger.error(f"Error during session rollback: {rollback_error}")
        logger.error(f"Database session error during yield: {e}", exc_info=True)
        raise
    finally:
        try:
            await session.close()
        except Exception as close_error:
            logger.error(f"Error closing database session: {close_error}")
