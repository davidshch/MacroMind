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

    async def predict_volatility(self, historical_data: List[float], current_sentiment_score: float) -> float:
        """Predict future volatility based on price history and current sentiment.

        Args:
            historical_data: List of historical closing prices.
            current_sentiment_score: The latest aggregated sentiment score (-1 to 1).

        Returns:
            Predicted volatility score.
        """
        try:
            if not historical_data or len(historical_data) < 21: # Need enough data for rolling features
                 logger.warning("Not enough historical data for volatility prediction.")
                 return 0.0 # Or raise an error

            df = pd.DataFrame(historical_data, columns=['price'])
            df['returns'] = df['price'].pct_change()
            # Calculate rolling volatility (standard deviation of returns)
            df['vol_rolling_std'] = df['returns'].rolling(window=20).std() * np.sqrt(252) # Annualized

            # Drop NaNs created by pct_change and rolling
            df.dropna(inplace=True)

            if df.empty:
                logger.warning("DataFrame empty after calculating features for volatility prediction.")
                return 0.0

            # Prepare features for the last data point
            last_features = df[['vol_rolling_std']].iloc[-1:].copy() # Use rolling std dev as primary feature
            # Add the sentiment score as an additional feature
            last_features['sentiment'] = current_sentiment_score

            # Ensure the model is trained/fitted appropriately before prediction
            # NOTE: This assumes the self.volatility_model (XGBRegressor) is already trained
            # with features named 'vol_rolling_std' and 'sentiment'.
            # Training logic is not shown here but is crucial.
            # If the model is not trained, prediction will fail or be meaningless.
            # Example check (replace with actual model state check):
            # if not hasattr(self.volatility_model, '_Booster'):
            #     logger.error("Volatility model is not trained!")
            #     # Handle untrained model case: train it or return default/error
            #     # For now, we proceed assuming it's trained.
            #     # You might need to implement lazy loading/training here.
            #     return 0.0 # Placeholder for untrained model

            # Predict using the last available features + sentiment
            # Ensure the feature names match exactly what the model was trained on
            prediction = self.volatility_model.predict(last_features)
            return float(prediction[0])

        except xgb.core.XGBoostError as xgb_err:
             logger.error(f"XGBoost error during volatility prediction: {xgb_err}")
             # Check if it's a feature mismatch error
             if "feature_names mismatch" in str(xgb_err):
                 logger.error("Feature names mismatch. Model trained features vs. prediction features.")
                 logger.error(f"Model expected features: {self.volatility_model.get_booster().feature_names}")
                 logger.error(f"Provided features: {last_features.columns.tolist()}")
             return 0.0 # Return default on error
        except Exception as e:
            logger.error(f"Volatility prediction error: {str(e)}")
            return 0.0 # Return default on error

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
