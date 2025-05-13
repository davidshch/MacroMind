"""
Main entry point for MacroMind API.
Handles market data streaming, sentiment analysis, and user authentication.
"""

from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import logging
import logging.config
from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .config import get_settings, Settings
from .database.database import engine, SessionLocal, get_db
from .api.routes import (
    auth,
    market_data,
    sentiment,
    volatility,
    vip,
    sector_data,
    admin,
    alerts,
    economic_calendar,
    opportunities
)
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.security import SecurityHeadersMiddleware

from .services.alerts import AlertService
from .services.sentiment_analysis import SentimentAnalysisService
from .services.volatility import VolatilityService
from .services.market_data import MarketDataService
from .services.ml.model_factory import MLModelFactory

from .core.dependencies import (
    get_ml_model_factory,
    get_market_data_service,
)

# --- Settings --- #
settings: Settings = get_settings()

# --- Logging Configuration --- #
log_level_str = settings.log_level.upper()
log_level_int = getattr(logging, log_level_str, logging.INFO)
if not isinstance(getattr(logging, log_level_str, None), int):
    print(f"Warning: Invalid LOG_LEVEL '{settings.log_level}'. Defaulting to INFO.")
    log_level_int = logging.INFO
    log_level_str = "INFO"

LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': log_level_int,
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': log_level_int,
            'propagate': False
        },
        'uvicorn.error': {
            'level': log_level_int,
            'handlers': ['console'],
            'propagate': False,
        },
        'uvicorn.access': {
            'handlers': ['console'],
            'level': logging.WARNING,
            'propagate': False,
        },
        'sqlalchemy.engine': {
            'handlers': ['console'],
            'level': logging.WARNING,
            'propagate': False,
        },
        'apscheduler': {
            'handlers': ['console'],
            'level': logging.INFO,
            'propagate': False,
        }
    }
}

try:
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured successfully. Application log level set to: {log_level_str}")
except Exception as e:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    logger.error(f"Error configuring logging with dictConfig: {e}. Using basicConfig.", exc_info=True)

# --- Scheduler Setup --- #
scheduler = AsyncIOScheduler()

async def scheduled_check_alerts():
    """
    Scheduled job to check and trigger alerts.
    Manually creates dependencies as it runs outside HTTP request scope.
    """
    logger.info("Scheduler: Starting scheduled_check_alerts job.")
    db_session: Optional[AsyncSession] = None
    try:
        db_session = SessionLocal()

        ml_factory = get_ml_model_factory()
        market_data_service = get_market_data_service(db=db_session)

        volatility_service = VolatilityService(db=db_session, market_data_service=market_data_service, ml_model_factory=ml_factory)
        sentiment_service = SentimentAnalysisService(db=db_session, market_data_service=market_data_service, ml_model_factory=ml_factory, volatility_service=volatility_service)
        alert_service = AlertService(db=db_session, sentiment_service=sentiment_service, volatility_service=volatility_service)

        await alert_service.check_and_trigger_alerts()
        logger.info("Scheduler: Finished scheduled_check_alerts job successfully.")
    except Exception as e:
        logger.error(f"Scheduler: Error during scheduled_check_alerts: {e}", exc_info=True)
    finally:
        if db_session:
            try:
                await db_session.close()
                logger.debug("Scheduler: Job DB session closed.")
            except Exception as close_err:
                logger.error(f"Scheduler: Error closing DB session in scheduled job: {close_err}", exc_info=True)

# --- FastAPI Lifespan Event Handler --- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Initializing scheduler...")
    alert_check_interval = getattr(settings, 'alert_check_interval_minutes', 1)
    scheduler.add_job(
        scheduled_check_alerts,
        'interval',
        minutes=alert_check_interval,
        id="check_alerts_job",
        max_instances=1,
        misfire_grace_time=60
    )
    scheduler.start()
    logger.info(f"Scheduler started. Alert check interval: {alert_check_interval} minutes.")
    yield
    logger.info("Application shutdown: Shutting down scheduler...")
    scheduler.shutdown()
    logger.info("Scheduler shut down.")

# --- App Initialization --- #
app = FastAPI(
    title="MacroMind API",
    description="API for MacroMind, an AI-powered economic calendar and financial analytics platform.",
    version="0.1.0",
    lifespan=lifespan
)

# --- Middleware --- #
allowed_origins_config = getattr(settings, 'allowed_origins', [])
if not allowed_origins_config:
    logger.warning("ALLOWED_ORIGINS not set in .env, defaulting to empty list (no CORS). For development, consider ['http://localhost:3000'] or similar.")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins_config,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"CORS Middleware added. Allowed origins: {allowed_origins_config}")

app.add_middleware(SecurityHeadersMiddleware)
logger.info("Security Headers Middleware added.")

use_redis_for_rate_limit = getattr(settings, 'use_redis', False)
if use_redis_for_rate_limit:
    try:
        app.add_middleware(RateLimitMiddleware)
        logger.info("Rate Limiting Middleware added (Redis enabled).")
    except Exception as e:
        logger.error(f"Failed to add RateLimitMiddleware: {e}. Check Redis connection and middleware setup.", exc_info=True)
else:
    logger.info("Rate Limiting Middleware skipped (use_redis is false or not set).")

# --- API Routers --- #
logger.info("Including API routers...")
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(market_data.router, prefix="/api/market", tags=["Market Data"])
app.include_router(sentiment.router, prefix="/api/sentiment", tags=["Sentiment Analysis"])
app.include_router(volatility.router, prefix="/api/volatility", tags=["Volatility Analysis"])
app.include_router(sector_data.router, prefix="/api/sectors", tags=["Sector Data"])
app.include_router(vip.router, prefix="/api/v1/ai", tags=["AI (VIP)"])
app.include_router(admin.router, prefix="/api/admin", tags=["Admin"])
app.include_router(alerts.router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(economic_calendar.router, prefix="/api/calendar", tags=["Economic Calendar"])
app.include_router(opportunities.router, prefix="/api/opportunities", tags=["Market Opportunities"])
logger.info("API routers included.")

# --- Root Endpoint --- #
@app.get("/")
async def read_root():
    """Provides basic API information."""
    logger.info("Root endpoint / accessed.")
    return {"message": "Welcome to the MacroMind API", "version": app.version}

# --- Health Check Endpoint --- #
@app.get("/health", tags=["Health"])
async def health_check(db: AsyncSession = Depends(get_db)):
    """Performs a health check on the API and database connection."""
    logger.debug("Health check endpoint /health accessed.")
    db_status = "error"
    try:
        result = await db.execute(select(1))
        if result.scalar_one() == 1:
            db_status = "ok"
        else:
            logger.warning("Health check: DB query did not return 1.")
    except Exception as e:
        logger.error(f"Health check: Database connection error: {e}", exc_info=True)

    return {"status": "ok", "database_status": db_status, "timestamp": datetime.now().isoformat()}

logger.info("FastAPI app initialization complete. Waiting for Uvicorn to start server...")