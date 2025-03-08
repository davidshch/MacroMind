from fastapi import FastAPI
from .api.routes import auth, events, market_data, sentiment, volatility, economic_calendar, vip, admin

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

@app.get("/")
async def read_root():
    return {"message": "Welcome to MacroMind API!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}