"""
Main entry point for MacroMind API.
Handles market data streaming, sentiment analysis, and user authentication.
"""

from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
import logging
import logging.config
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import select

from .config import get_settings
from .database.database import engine, Base, SessionLocal, get_db
from .api.routes import (
    auth, 
    market_data, 
    sentiment, 
    volatility, 
    vip
)
from .middleware.rate_limit import RateLimitMiddleware
from .middleware.security import SecurityHeadersMiddleware

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

# --- App Initialization --- #
settings = get_settings()

app = FastAPI(
    title="MacroMind API",
    description="API for MacroMind financial analytics platform.",
    version="0.1.0"
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

# --- Database Initialization --- #
@app.on_event("startup")
async def startup_event():
    logger.info("Executing startup event...")
    try:
        async with engine.begin() as conn:
            logger.info("Database connection successful. (Alembic manages schema)")
        logger.info("WebSocket manager initialized.")
    except Exception as e:
        logger.error(f"Database connection failed during startup: {e}")
    logger.info("Startup event completed.")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Executing shutdown event...")
    await engine.dispose()
    logger.info("Database engine disposed.")
    logger.info("Shutdown event completed.")

# --- API Routers --- #
logger.info("Including API routers...")
app.include_router(auth.router)
app.include_router(market_data.router)
app.include_router(sentiment.router)
app.include_router(volatility.router)
app.include_router(vip.router)
logger.info("API routers included.")

# --- Root Endpoint --- #
@app.get("/")
async def read_root():
    logger.info("Root endpoint accessed.")
    return {"message": "Welcome to the MacroMind API"}

# --- Health Check Endpoint --- #
@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    logger.debug("Health check endpoint accessed.")
    try:
        db.execute(select(1))
        db_status = "OK"
    except Exception as e:
        logger.error(f"Health check DB connection failed: {e}")
        db_status = "Error"
    
    return {"status": "OK", "database": db_status, "timestamp": datetime.now().isoformat()}