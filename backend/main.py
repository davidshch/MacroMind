from fastapi import FastAPI
from backend.market_data import get_economic_events, save_economic_events

app = FastAPI()

@app.get("/")
def home():
    return {"message": "MacroMind API is running!"}

@app.get("/economic-events")
def get_economic_events():
    return {"events": "List of economic events"}

@app.get("/fetch-and-save-events")
def fetch_and_save_events():
    events = get_economic_events()  # Fetch from API
    save_economic_events(events)    # Store in PostgreSQL
    return {"message": "Events saved successfully!"}