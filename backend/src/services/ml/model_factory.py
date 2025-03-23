from transformers import pipeline
from prophet import Prophet
import xgboost as xgb
from typing import Dict, Any, List
import numpy as np
from datetime import datetime
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class MLModelFactory:
    """Factory for sentiment, volatility, and forecasting models."""
    
    def __init__(self):
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            return_all_scores=True
        )
        
        self.prophet = Prophet(daily_seasonality=True)
        self.volatility_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            n_estimators=100
        )

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze text sentiment using FinBERT.
        
        Returns:
            dict: Sentiment ("bullish"/"bearish") with confidence score
        """
        try:
            results = self.sentiment_pipeline(text)[0]
            sentiment = max(results, key=lambda x: x['score'])
            return {
                "sentiment": "bullish" if sentiment["label"] == "positive" else "bearish",
                "confidence": float(sentiment["score"]),
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return {"sentiment": "neutral", "confidence": 0.0}

    async def predict_volatility(self, historical_data: List[float]) -> float:
        """Predict future volatility based on price history."""
        try:
            df = pd.DataFrame(historical_data, columns=['price'])
            df['returns'] = df['price'].pct_change()
            df['vol'] = df['returns'].rolling(20).std()
            
            features = df[['returns', 'vol']].dropna().values
            prediction = self.volatility_model.predict(features[-1:])
            return float(prediction[0])
        except Exception as e:
            logger.error(f"Volatility prediction error: {str(e)}")
            return 0.0

    async def forecast_prices(self, dates: List[str], prices: List[float]) -> Dict[str, Any]:
        """Generate 7-day price forecast using Prophet."""
        try:
            df = pd.DataFrame({
                'ds': pd.to_datetime(dates),
                'y': prices
            })
            self.prophet.fit(df)
            future = self.prophet.make_future_dataframe(periods=7)
            forecast = self.prophet.predict(future)
            
            return {
                "forecast": forecast['yhat'].tail(7).tolist(),
                "dates": forecast['ds'].tail(7).dt.strftime('%Y-%m-%d').tolist()
            }
        except Exception as e:
            logger.error(f"Price forecasting error: {str(e)}")
            return {"forecast": [], "dates": []}
