"""
Main entry point for MacroMind API.
Handles market data streaming, sentiment analysis, and user authentication.
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
import logging.config
from datetime import datetime
from typing import Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from contextlib import asynccontextmanager
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from .config import get_settings
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
    get_sentiment_service,
    get_volatility_service
)

# --- Logging Configuration --- #
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        },
    },
    'handlers': {
        'console': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
            'stream': 'ext://sys.stdout',
        },
    },
    'loggers': {
        '': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False
        },
        'uvicorn.error': {
            'level': 'INFO',
            'handlers': ['console'],
            'propagate': False,
        },
        'uvicorn.access': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
        'sqlalchemy.engine': {
             'handlers': ['console'],
             'level': 'WARNING',
             'propagate': False,
        },
        'apscheduler': {
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        }
    }
}

try:
    logging.config.dictConfig(LOGGING_CONFIG)
    logger = logging.getLogger(__name__)
    logger.info("Logging configured successfully.")
except Exception as e:
    print(f"Error configuring logging: {e}")
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.warning("Using basic logging configuration due to error.")

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
        market_data_serv = get_market_data_service()
        
        vol_service = VolatilityService(
            db=db_session, 
            market_data_service=market_data_serv, 
            ml_model_factory=ml_factory
        )
        
        sent_service = SentimentAnalysisService(
            db=db_session, 
            market_data_service=market_data_serv, 
            ml_model_factory=ml_factory,
            volatility_service=vol_service
        )
        
        alert_serv = AlertService(
            db=db_session,
            sentiment_service=sent_service,
            volatility_service=vol_service
        )
        
        await alert_serv.check_and_trigger_alerts()
        logger.info("Scheduler: Finished scheduled_check_alerts job.")
    except Exception as e:
        logger.error(f"Scheduler: Error during scheduled_check_alerts: {e}", exc_info=True)
    finally:
        if db_session:
            await db_session.close()
            logger.debug("Scheduler: Job DB session closed.")

# --- FastAPI Lifespan Event Handler --- #
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Application startup: Initializing scheduler...")
    scheduler.add_job(scheduled_check_alerts, 'interval', minutes=1, id="check_alerts_job")
    scheduler.start()
    logger.info("Scheduler started with job 'check_alerts_job'.")
    yield
    logger.info("Application shutdown: Shutting down scheduler...")
    scheduler.shutdown()
    logger.info("Scheduler shut down.")

# --- App Initialization --- #
settings = get_settings()

app = FastAPI(
    title="MacroMind API",
    description="API for MacroMind financial analytics platform.",
    version="0.1.0",
    lifespan=lifespan
)

# --- Middleware --- #
logger.info("Adding Middleware...")
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info(f"CORS enabled for origins: {settings.allowed_origins}")

app.add_middleware(SecurityHeadersMiddleware)
logger.info("Security Headers Middleware added.")

if settings.use_redis:
    app.add_middleware(RateLimitMiddleware)
    logger.info("Rate Limiting Middleware added (Redis enabled).")
else:
    logger.info("Rate Limiting Middleware skipped (Redis not enabled).")

# --- API Routers --- #
logger.info("Including API routers...")
app.include_router(auth.router)
app.include_router(market_data.router)
app.include_router(sentiment.router)
app.include_router(volatility.router)
app.include_router(sector_data.router)
app.include_router(vip.router)
app.include_router(admin.router)
app.include_router(alerts.router, prefix="/api/v1/alerts", tags=["Alerts"])

if hasattr(economic_calendar, 'router'):
    if not any(r.path == economic_calendar.router.prefix for r in app.routes if hasattr(r, 'path')):
        app.include_router(economic_calendar.router, prefix="/api/v1/calendar", tags=["Economic Calendar"]) 
    else:
        logger.info("Economic calendar router already included or prefix mismatch.")
else:
    logger.warning("economic_calendar.router not found, skipping inclusion.")

app.include_router(opportunities.router, prefix="/api/v1/opportunities", tags=["Market Opportunities"])
logger.info("API routers included.")

# --- Root Endpoint --- #
@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the MacroMind API"}

# --- Health Check Endpoint --- #
@app.get("/health")
async def health_check(db: AsyncSession = Depends(get_db)):
    logger.debug("Health check endpoint accessed.")
    try:
        await db.execute(select(1))
        db_status = "OK"
    except Exception as e:
        logger.error(f"Health check DB connection failed: {e}")
        db_status = "Error"
    
    return {"status": "OK", "database": db_status, "timestamp": datetime.now().isoformat()}