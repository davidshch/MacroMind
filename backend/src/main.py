"""
Main entry point for MacroMind API.
Handles market data streaming, sentiment analysis, and user authentication.
"""

from fastapi import FastAPI
from .api.routes import auth, events, market_data, sentiment, volatility, economic_calendar, vip, admin, alerts, visualization, websocket
import asyncio
from .services.websocket import ConnectionManager

app = FastAPI(
    title="MacroMind API",
    description="""
    MacroMind is an advanced AI-driven economic calendar and market analysis platform.
    It provides real-time market insights, sentiment analysis, and volatility predictions
    for traders, investors, and financial analysts.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Include the API routes
app.include_router(auth.router)
app.include_router(events.router)
app.include_router(market_data.router)
app.include_router(sentiment.router)
app.include_router(volatility.router)
app.include_router(economic_calendar.router)
app.include_router(vip.router)
app.include_router(admin.router)
app.include_router(alerts.router)
app.include_router(visualization.router)
app.include_router(websocket.router)

@app.on_event("startup")
async def start_streaming():
    """Start WebSocket streaming on app startup."""
    manager = ConnectionManager()
    asyncio.create_task(manager.start_streaming())

@app.get("/")
async def read_root():
    """Root endpoint with API status."""
    return {
        "name": "MacroMind API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs"
    }

@app.get("/health")
async def health_check():
    """Simple health check endpoint."""
    return {"status": "healthy"}