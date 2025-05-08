import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
import re
import os
import sys
import logging
from datetime import datetime
from pathlib import Path

# Set up testing environment variables if not already set
os.environ['TESTING'] = 'TRUE'

# Configure logging for tests to reduce noise
logging.basicConfig(level=logging.INFO)
for noisy_logger in ['sqlalchemy.engine', 'httpx', 'asyncio']:
    logging.getLogger(noisy_logger).setLevel(logging.WARNING)

# Import Base from the correct location
from src.database.models import Base
from src.database.database import get_db
from dotenv import load_dotenv

# Load environment variables from .env
try:
    env_path = Path('.env')
    if env_path.exists():
        load_dotenv(dotenv_path=str(env_path), override=True)
        print(f"Loaded environment variables from {env_path}")
    else:
        print("Warning: .env file not found, using environment variables")
except Exception as e:
    print(f"Error loading .env file: {e}")
    # Continue with environment variables

# Use a separate test database if configured, otherwise use main DB URL
DATABASE_URL = os.getenv("DATABASE_URL")
TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL", DATABASE_URL)

if not TEST_DATABASE_URL:
    # Fallback to a SQLite in-memory database for testing if no DB URL is provided
    print("No database URL provided, using in-memory SQLite for testing")
    TEST_DATABASE_URL = "sqlite:///./test.db" 

# Create a synchronous engine URL for setup/teardown
SYNC_TEST_DATABASE_URL = TEST_DATABASE_URL
if SYNC_TEST_DATABASE_URL and SYNC_TEST_DATABASE_URL.startswith("postgresql+asyncpg"):
    SYNC_TEST_DATABASE_URL = re.sub(r"^postgresql\+asyncpg", "postgresql+psycopg2", SYNC_TEST_DATABASE_URL)
elif SYNC_TEST_DATABASE_URL and SYNC_TEST_DATABASE_URL.startswith("sqlite+aiosqlite"):
    # For sqlite, the sync version is just sqlite:///
    SYNC_TEST_DATABASE_URL = re.sub(r"^sqlite+aiosqlite", "sqlite", SYNC_TEST_DATABASE_URL)

print(f"Using test database: {TEST_DATABASE_URL}")
print(f"Using sync database for setup: {SYNC_TEST_DATABASE_URL}")

# Use a simple in-memory SQLite engine if needed for quick testing
# Comment out the following lines when testing with a real database
if not os.getenv("USE_REAL_DB"):
    print("Using in-memory SQLite database for faster testing")
    SYNC_TEST_DATABASE_URL = "sqlite:///:memory:"
    TEST_DATABASE_URL = "sqlite:///:memory:"

# Use a synchronous engine for creating tables
try:
    sync_engine = create_engine(SYNC_TEST_DATABASE_URL, connect_args={"connect_timeout": 5})
    # Apply migrations or create tables for the test database
    # Base.metadata.drop_all(bind=sync_engine) # Optional: Drop tables first
    Base.metadata.create_all(bind=sync_engine)
    print("Test database tables created successfully")
except Exception as e:
    print(f"Warning: Error creating test database tables with sync engine: {e}")
    # Continue with testing even if table creation fails
    # Tests that need DB tables will fail, but others might succeed

# Use the potentially async engine for the actual test sessions if needed by the app
# If your app uses async sessions, this engine should match.
# If tests are purely synchronous, using sync_engine might be simpler.
engine = create_engine(TEST_DATABASE_URL, connect_args={"connect_timeout": 5})
TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@pytest.fixture(scope="session")
def db_engine():
    """Yields a SQLAlchemy engine instance (potentially async)."""
    return engine

@pytest.fixture(scope="function")
def db_session(db_engine) -> Generator[Session, None, None]:
    """Yields a SQLAlchemy session bound to a transaction for testing."""
    connection = db_engine.connect()
    # begin a non-ORM transaction
    trans = connection.begin()
    # bind an ORM session to the connection
    session = TestingSessionLocal(bind=connection)

    try:
        yield session
    finally:
        session.close()
        # rollback - everything that happened with the
        # session above (including calls to commit())
        # is rolled back.
        trans.rollback()
        connection.close()


@pytest.fixture(scope="function")
def test_client(db_session) -> Generator[TestClient, None, None]:
    """Creates a test client with the session override."""
    # Import here to prevent circular imports
    from src.main import app

    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    
    with TestClient(app) as client:
        yield client
    
    # Clean up the override
    app.dependency_overrides.clear()


@pytest.fixture(scope="session")
def mock_settings():
    """Provides test settings with reasonable defaults."""
    from src.config import Settings
    
    return Settings(
        database_url=TEST_DATABASE_URL,
        api_key_alpha_vantage="test_av_key",
        finnhub_api_key="test_fh_key",
        jwt_secret="test_secret",
        jwt_algorithm="HS256",
        access_token_expire_minutes=30,
        reddit_client_id="test_reddit_id",
        reddit_client_secret="test_reddit_secret",
        use_demo_data=True
    )
