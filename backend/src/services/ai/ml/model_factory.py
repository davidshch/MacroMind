import asyncio
import logging
import os
from typing import Dict, List, Any, Union, Optional

import joblib
import numpy as np
import pandas as pd
from prophet import Prophet
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    pipeline,
)
from xgboost import XGBRegressor

# Configure logging
logger = logging.getLogger(__name__)


class MLModelFactory:
    """
    A factory class for creating and using various machine learning models.
    Provides methods for sentiment analysis, price forecasting, and volatility prediction.
    """
    # Define the directory for storing trained models relative to this file
    TRAINED_MODELS_DIR = os.path.join(os.path.dirname(__file__), "trained_models")

    def __init__(self):
        """
        Initializes the MLModelFactory.
        Currently, pre-loads the FinBERT sentiment analysis pipeline.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.sentiment_pipeline = None
        try:
            # Initialize FinBERT pipeline
            # Using specific model and tokenizer for more control.
            # Fallback to simpler pipeline creation if specific model loading fails.
            model_name = "ProsusAI/finbert"
            try:
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSequenceClassification.from_pretrained(model_name)
                self.sentiment_pipeline = pipeline(
                    "sentiment-analysis", model=model, tokenizer=tokenizer
                )
                self.logger.info(
                    f"FinBERT sentiment pipeline initialized successfully with {model_name}."
                )
            except OSError: # Handles model not found locally / network issues for specific loading
                self.logger.warning(
                    f"Could not load {model_name} directly. Attempting generic pipeline init."
                )
                self.sentiment_pipeline = pipeline("sentiment-analysis", model=model_name)
                self.logger.info(
                    f"FinBERT sentiment pipeline initialized successfully using generic pipeline(model='{model_name}')."
                )

        except Exception as e:
            self.logger.error(
                f"Failed to initialize FinBERT sentiment pipeline: {e}", exc_info=True
            )
            # self.sentiment_pipeline remains None, methods should handle this

    async def analyze_sentiment(self, text: str) -> List[Dict[str, Any]]:
        """
        Analyzes sentiment of a given text using the pre-loaded FinBERT pipeline.

        Args:
            text: The input text to analyze.

        Returns:
            A list of dictionaries, where each dictionary contains 'label'
            (e.g., 'positive', 'negative', 'neutral') and 'score' (confidence).
            Returns an error message structure if analysis fails or model is unavailable.
            Example: [{'label': 'positive', 'score': 0.98}]
        """
        if self.sentiment_pipeline is None:
            self.logger.error(
                "Sentiment analysis pipeline is not available. Check initialization."
            )
            return [
                {
                    "label": "error",
                    "score": 0.0,
                    "message": "Sentiment model not loaded",
                }
            ]

        if not text or not isinstance(text, str) or not text.strip():
            self.logger.warning(
                "Empty or invalid text provided for sentiment analysis."
            )
            return [] # Or a specific error format: [{"label": "error", "score": 0.0, "message": "Empty input text"}]


        try:
            # Transformers pipeline is synchronous, run in a thread pool for async compatibility
            loop = asyncio.get_running_loop()
            results = await loop.run_in_executor(None, self.sentiment_pipeline, text)
            # Ensure results are serializable and in the expected format
            # FinBERT pipeline usually returns List[Dict[str, Union[str, float]]]
            # e.g. [{'label': 'positive', 'score': 0.9873789548873901}]
            return results if isinstance(results, list) else [results]
        except Exception as e:
            self.logger.error(
                f"Error during sentiment analysis for text '{text[:100]}...': {e}",
                exc_info=True,
            )
            return [
                {"label": "error", "score": 0.0, "message": f"Analysis failed: {str(e)}"}
            ]

    async def forecast_prices(
        self,
        dates: List[Any],  # Can be str, datetime objects
        prices: List[float],
        periods: int,
        freq: str = "D",  # 'D' for daily, 'B' for business day, 'H' for hourly etc.
        growth: str = "linear",  # 'linear' or 'logistic'
        capacity_df: Optional[pd.DataFrame] = None,  # For logistic growth: df with 'ds' and 'cap'
        **kwargs: Any  # Additional Prophet params (e.g., seasonality_mode)
    ) -> Dict[str, Any]:
        """
        Forecasts future prices using Facebook Prophet.

        Args:
            dates: List of historical dates (strings, datetime objects).
            prices: List of corresponding historical prices.
            periods: Number of periods to forecast into the future.
            freq: Frequency of the forecast periods (e.g., 'D', 'B', 'H', 'M').
            growth: Type of growth model ('linear' or 'logistic').
                    If 'logistic', `capacity_df` (or 'cap' in historical df) is required.
            capacity_df: Optional DataFrame with 'ds' and 'cap' columns for logistic growth.
                         'ds' should cover historical and future dates.
            **kwargs: Additional arguments to pass to the Prophet model constructor
                      (e.g., yearly_seasonality, weekly_seasonality, daily_seasonality, seasonality_mode).

        Returns:
            A dictionary containing the forecast DataFrame (as a list of dicts)
            and parameters used. Returns an error structure on failure.
            Example: {'forecast_df': [{'ds': '2023-01-01', 'yhat': 150.0, ...}],
                      'params': {'periods': 30, 'freq': 'D', 'growth': 'linear'}}
        """
        if not dates or not prices:
            self.logger.warning("Empty dates or prices list provided for forecasting.")
            return {"error": "Dates and prices cannot be empty."}
        if len(dates) != len(prices):
            self.logger.error("Dates and prices lists must have the same length.")
            return {"error": "Dates and prices lists length mismatch."}

        df = pd.DataFrame({"ds": pd.to_datetime(dates), "y": prices})

        if growth == "logistic":
            if "cap" not in df.columns:
                if capacity_df is not None and not capacity_df.empty:
                    # Ensure capacity_df 'ds' is datetime
                    capacity_df['ds'] = pd.to_datetime(capacity_df['ds'])
                    df = pd.merge(df, capacity_df[['ds', 'cap']], on="ds", how="left")
                else:
                    self.logger.error("For logistic growth, 'cap' column must be in historical data or provided via capacity_df.")
                    return {"error": "Capacity data ('cap') required for logistic growth."}
            
            if df["cap"].isnull().any():
                self.logger.warning("Missing 'cap' values for logistic growth. Attempting ffill/bfill.")
                df["cap"] = df["cap"].ffill().bfill()
                if df["cap"].isnull().any():
                    self.logger.error("Still missing 'cap' values after fill. Cannot proceed with logistic growth.")
                    return {"error": "Incomplete capacity data for logistic growth."}


        try:
            model = Prophet(growth=growth, **kwargs)
            
            loop = asyncio.get_running_loop()
            # Fit model in executor
            await loop.run_in_executor(None, model.fit, df)

            # Make future dataframe in executor
            future_df = await loop.run_in_executor(None, model.make_future_dataframe, periods, freq)

            if growth == "logistic":
                if "cap" not in future_df.columns: # If 'cap' was not part of df initially or make_future_dataframe didn't pick it up
                    if capacity_df is not None and not capacity_df.empty:
                        # Merge with full capacity_df (historical + future)
                        future_df = pd.merge(future_df, capacity_df[['ds', 'cap']], on="ds", how="left")
                    elif 'cap' in df.columns: # Use last known cap from historical data
                         future_df['cap'] = df['cap'].iloc[-1] # Simplistic: assumes constant future capacity

                if future_df["cap"].isnull().any():
                    self.logger.warning("Missing 'cap' values in future_df for logistic growth. Attempting ffill/bfill from last known value.")
                    future_df["cap"] = future_df["cap"].ffill().bfill() # Fill from historical, then future if any
                    if future_df["cap"].isnull().any(): # If still null, use last historical value
                        last_hist_cap = df['cap'].iloc[-1] if 'cap' in df.columns and not df['cap'].empty else None
                        if last_hist_cap is not None:
                            future_df['cap'].fillna(last_hist_cap, inplace=True)
                        else:
                            self.logger.error("Cannot determine future capacity for logistic growth.")
                            return {"error": "Future capacity ('cap') could not be determined for logistic growth."}


            # Predict in executor
            forecast_df = await loop.run_in_executor(None, model.predict, future_df)

            return {
                "forecast_df": forecast_df[
                    ["ds", "yhat", "yhat_lower", "yhat_upper", "trend", "trend_lower", "trend_upper"]
                ].to_dict(orient="records"),
                "params": {"periods": periods, "freq": freq, "growth": growth},
            }
        except Exception as e:
            self.logger.error(f"Error during Prophet price forecasting: {e}", exc_info=True)
            return {"error": f"Prophet forecasting failed: {str(e)}"}

    async def train_volatility_model(
        self,
        features_df: pd.DataFrame,
        target_series: pd.Series,
        save_path: str,
        model_params: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Trains an XGBoost regression model for volatility and saves it.

        Args:
            features_df: DataFrame containing the training features.
            target_series: Series containing the target variable.
            save_path: Absolute path where the trained model should be saved.
            model_params: Optional dictionary of parameters for XGBRegressor.

        Returns:
            True if training and saving were successful, False otherwise.
        """
        if features_df.empty or target_series.empty:
            self.logger.error("Cannot train volatility model: Features or target series is empty.")
            return False
        
        if len(features_df) != len(target_series):
            self.logger.error("Feature and target series length mismatch.")
            return False

        # Ensure the directory for saving the model exists
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        except Exception as e:
            self.logger.error(f"Error creating directory {os.path.dirname(save_path)}: {e}", exc_info=True)
            return False

        default_params = {
            'objective': 'reg:squarederror',
            'n_estimators': 100,
            'learning_rate': 0.05,
            'max_depth': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'tree_method': 'hist' # Faster for larger datasets
        }
        if model_params:
            default_params.update(model_params)

        model = XGBRegressor(**default_params)

        try:
            self.logger.info(f"Starting XGBoost model training. Features: {features_df.columns.tolist()}, Target: {target_series.name}")
            loop = asyncio.get_running_loop()
            # XGBoost fit is CPU-bound, run in executor
            await loop.run_in_executor(None, model.fit, features_df, target_series)
            self.logger.info(f"XGBoost model training completed. Saving model to {save_path}")
            
            # joblib.dump is I/O bound, can also be in executor
            await loop.run_in_executor(None, joblib.dump, model, save_path)
            self.logger.info(f"Model successfully saved to {save_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error during volatility model training or saving: {e}", exc_info=True)
            return False

    async def predict_volatility(
        self, model_path: str, features_df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Predicts volatility using a pre-trained XGBoost model.

        Args:
            model_path: Absolute path to the saved XGBoost model file (e.g., .joblib or .xgb).
            features_df: DataFrame containing the features for prediction.
                         The columns must match the features and order used during training.

        Returns:
            A dictionary containing the prediction(s), model_path, feature_names,
            or an error message.
        """
        if not os.path.isabs(model_path):
            self.logger.warning(f"Model path '{model_path}' is not absolute. This might lead to issues.")

        if not os.path.exists(model_path):
            self.logger.error(f"Volatility model not found at path: {model_path}")
            return {"error": "Volatility model file not found", "model_path": model_path}

        try:
            loop = asyncio.get_running_loop()
            # Load model in executor
            model: XGBRegressor = await loop.run_in_executor(None, joblib.load, model_path)

            if not isinstance(model, XGBRegressor):
                self.logger.error(
                    f"Loaded object from {model_path} is not an XGBRegressor instance. Type: {type(model)}"
                )
                return {
                    "error": "Invalid model type loaded. Expected XGBRegressor.",
                    "model_path": model_path,
                }

            # Predict in executor
            raw_predictions = await loop.run_in_executor(None, model.predict, features_df)

            feature_names = []
            if hasattr(model, 'feature_names_in_'):
                feature_names = list(model.feature_names_in_)
            elif hasattr(model, 'get_booster') and hasattr(model.get_booster(), 'feature_names'):
                 feature_names = model.get_booster().feature_names

            # model.predict returns a numpy array.
            if isinstance(raw_predictions, np.ndarray):
                predictions_list = raw_predictions.tolist()
                volatility_prediction = predictions_list[0] if len(predictions_list) == 1 else predictions_list
            else:
                volatility_prediction = float(raw_predictions)

            return {
                "prediction": volatility_prediction, 
                "model_path": model_path,
                "feature_names": feature_names
            }
        except FileNotFoundError: # Should be caught by os.path.exists, but as a safeguard
            self.logger.error(f"Volatility model file not found during load: {model_path}", exc_info=True)
            return {"error": "Volatility model file not found during load", "model_path": model_path}
        except Exception as e:
            # Ensure features_df.head() is converted to string for logging to avoid issues
            features_head_str = str(features_df.head())
            self.logger.error(
                f"Error during volatility prediction with model {model_path} for features:\n{features_head_str}",
                exc_info=True,
            )
            return {"error": f"Volatility prediction failed: {str(e)}", "model_path": model_path}

_ml_model_factory_instance = None

def get_ml_model_factory() -> MLModelFactory:
    """
    Dependency provider for MLModelFactory.
    
    Returns:
        MLModelFactory: A singleton instance of the model factory.
    """
    global _ml_model_factory_instance
    if _ml_model_factory_instance is None:
        _ml_model_factory_instance = MLModelFactory()
    return _ml_model_factory_instance
