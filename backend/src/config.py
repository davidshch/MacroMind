from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    app_name: str = "MacroMind API"
    database_url: str
    api_key_alpha_vantage: str
    api_key_news: str
    jwt_secret: str
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 30  # minutes
    # Add Reddit API settings
    reddit_client_id: str
    reddit_client_secret: str
    # Add Finnhub API settings
    finnhub_api_key: str  # Add Finnhub API key setting
    use_demo_data: bool = True  # Set to False in production

    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings():
    return Settings()
