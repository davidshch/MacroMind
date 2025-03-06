from fastapi import FastAPI
from .api.routes import auth, events, market_data, sentiment

app = FastAPI()

# Include the API routes
app.include_router(auth.router)
app.include_router(events.router)
app.include_router(market_data.router)
app.include_router(sentiment.router)

@app.get("/")
def read_root():
    return {"message": "Welcome to MacroMind API!"}