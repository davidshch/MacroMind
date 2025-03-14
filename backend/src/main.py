from fastapi import FastAPI
from .api.routes import auth, events, market_data, sentiment, volatility, economic_calendar, vip, admin, alerts, visualization, websocket
import asyncio
from .services.websocket import ConnectionManager

app = FastAPI(title="MacroMind API")

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
    """Start the WebSocket streaming on startup."""
    manager = ConnectionManager()
    asyncio.create_task(manager.start_streaming())

@app.get("/")
async def read_root():
    return {"message": "Welcome to MacroMind API!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}