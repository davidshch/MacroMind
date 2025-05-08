from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from typing import Optional
import logging

class Settings(BaseSettings):
    database_url: str = "sqlite:///./test.db"
    api_key_alpha_vantage: str = "demo"
    finnhub_api_key: str = "demo"
    api_key_news: Optional[str] = None
    jwt_secret: str = "test_secret"
    jwt_algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    reddit_client_id: Optional[str] = None
    reddit_client_secret: Optional[str] = None
    use_demo_data: bool = True

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="allow"
    )

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    try:
        return Settings()
    except Exception as e:
        logging.getLogger(__name__).error(f"Error loading settings: {e}")
        raise ValueError(f"Could not load application settings: {e}")
