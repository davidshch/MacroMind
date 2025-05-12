from transformers import pipeline
from prophet import Prophet
import xgboost as xgb
from typing import Dict, Any, List, Optional
import numpy as np
from datetime import datetime
import pandas as pd
import logging
import joblib
import os
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from ...config import get_settings

logger = logging.getLogger(__name__)

TRAINED_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')
DEFAULT_VOLATILITY_MODEL_PATH = os.path.join(TRAINED_MODELS_DIR, 'default_xgboost_volatility_model.joblib')

class MLModelFactory:
    """Factory for sentiment, volatility, and forecasting models."""
    
    def __init__(self, model_path: Optional[str] = None):
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
            return_all_scores=True
        )
        
        self.prophet = Prophet(daily_seasonality=True)
        
        self.volatility_model_path = model_path or DEFAULT_VOLATILITY_MODEL_PATH
        self.volatility_model = self._load_volatility_model(self.volatility_model_path)
        if self.volatility_model is None:
            logger.info(f"No pre-trained volatility model found at {self.volatility_model_path}. Initializing a new one.")
            self.volatility_model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                learning_rate=0.05,
                max_depth=5,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42
            )
            if os.path.exists(TRAINED_MODELS_DIR):
                self._save_volatility_model(self.volatility_model, self.volatility_model_path)
            else:
                logger.warning(f"Directory {TRAINED_MODELS_DIR} does not exist. Cannot save initial model.")

        # Initialize LLM client
        settings = get_settings()
        self.llm = None
        if settings.openai_api_key:
            try:
                self.llm = ChatOpenAI(openai_api_key=settings.openai_api_key, model_name="gpt-4-turbo") # Or "gpt-3.5-turbo"
                logger.info("ChatOpenAI LLM client initialized successfully.")
            except Exception as e:
                logger.error(f"Failed to initialize ChatOpenAI LLM client: {e}")
        else:
            logger.warning("OPENAI_API_KEY not found. LLM-based features will be disabled.")

    def _save_volatility_model(self, model: xgb.XGBRegressor, path: str) -> None:
        """Saves the volatility model to the specified path."""
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            joblib.dump(model, path)
            logger.info(f"Volatility model saved to {path}")
        except Exception as e:
            logger.error(f"Error saving volatility model to {path}: {e}")

    def _load_volatility_model(self, path: str) -> Optional[xgb.XGBRegressor]:
        """Loads the volatility model from the specified path."""
        if os.path.exists(path):
            try:
                model = joblib.load(path)
                logger.info(f"Volatility model loaded from {path}")
                return model
            except Exception as e:
                logger.error(f"Error loading volatility model from {path}: {e}")
                return None
        return None

    async def train_volatility_model(
        self, 
        features_df: pd.DataFrame, 
        target_series: pd.Series,
        save_path: Optional[str] = None
    ) -> bool:
        """
        Trains the XGBoost volatility model.

        Args:
            features_df: DataFrame with feature columns (e.g., historical_vol, sentiment, vix).
                         Column names must be consistent for future predictions.
            target_series: Series with the target variable (future realized volatility).
            save_path: Optional path to save the trained model. Defaults to self.volatility_model_path.

        Returns:
            True if training was successful, False otherwise.
        """
        if features_df.empty or target_series.empty:
            logger.error("Cannot train volatility model: Features DataFrame or target Series is empty.")
            return False
        
        if len(features_df) != len(target_series):
            logger.error("Cannot train volatility model: Features and target have different lengths.")
            return False

        original_columns = features_df.columns.tolist()
        safe_feature_names = [col.replace('[', '_').replace(']', '_').replace('<', '_') for col in original_columns]
        features_df.columns = safe_feature_names
        
        try:
            logger.info(f"Starting volatility model training with {len(features_df)} samples and features: {safe_feature_names}")
            self.volatility_model.fit(features_df, target_series)
            logger.info("Volatility model training completed.")
            
            if hasattr(self.volatility_model, 'get_booster'):
                self.volatility_model.get_booster().feature_names = safe_feature_names
            else:
                self.volatility_model.feature_names_in_ = safe_feature_names

            path_to_save = save_path or self.volatility_model_path
            self._save_volatility_model(self.volatility_model, path_to_save)
            return True
        except Exception as e:
            logger.exception(f"Error during volatility model training: {e}")
            return False

    async def analyze_sentiment(self, text: str) -> Dict[str, Any]:
        """Analyze text sentiment using FinBERT.
        
        Returns:
            dict: Contains 'primary_sentiment' (label), 'primary_confidence' (score),
                  'all_scores' (list of all label scores), and 'timestamp'.
        """
        try:
            # FinBERT pipeline is initialized with return_all_scores=True
            raw_results = self.sentiment_pipeline(text)
            if not raw_results or not raw_results[0]: # Ensure results are not empty
                raise ValueError("Sentiment pipeline returned empty or invalid results.")

            all_scores_list = raw_results[0] # This is a list of dicts like [{'label': 'positive', 'score': 0.9}, ...]
            
            if not all_scores_list: # Further check if the list itself is empty
                 raise ValueError("Sentiment pipeline returned an empty list of scores.")

            # Determine primary sentiment
            primary_sentiment_obj = max(all_scores_list, key=lambda x: x['score'])
            
            primary_label = "neutral" # Default
            if primary_sentiment_obj['label'].lower() == 'positive':
                primary_label = "bullish"
            elif primary_sentiment_obj['label'].lower() == 'negative':
                primary_label = "bearish"

            return {
                "primary_sentiment": primary_label,
                "primary_confidence": float(primary_sentiment_obj['score']),
                "all_scores": all_scores_list, # List of dicts: [{'label': 'positive', 'score': ...}, ...]
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Sentiment analysis error for text \"{text[:100]}...\": {str(e)}", exc_info=True)
            return {
                "primary_sentiment": "neutral", 
                "primary_confidence": 0.0,
                "all_scores": [],
                "error": str(e)
            }

    async def predict_volatility(self, features_for_prediction: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict future volatility using the trained XGBoost model.

        Args:
            features_for_prediction: DataFrame containing the features for a single prediction point.
                                     Must have the same columns (or a superset) as the training data,
                                     with original column names.

        Returns:
            A dictionary containing:
                - "prediction": The predicted volatility score (float).
                - "feature_names": A list of feature names used by the model (List[str]).
            Returns {"prediction": 0.0, "feature_names": []} if prediction fails.
        """
        if self.volatility_model is None:
            logger.error("Volatility model is not loaded.")
            return {"prediction": 0.0, "feature_names": []}

        # Check if the model is an XGBoost model and seems trained
        is_sklearn_wrapper = hasattr(self.volatility_model, 'feature_names_in_')
        is_native_xgboost = hasattr(self.volatility_model, 'get_booster') and hasattr(self.volatility_model.get_booster(), 'feature_names')

        if not (is_sklearn_wrapper or (is_native_xgboost and self.volatility_model.get_booster().feature_names)):
            if self.volatility_model_path == DEFAULT_VOLATILITY_MODEL_PATH and os.path.exists(DEFAULT_VOLATILITY_MODEL_PATH):
                logger.info(f"Default model at {DEFAULT_VOLATILITY_MODEL_PATH} seems untrained or lacks feature names. Attempting reload.")
                reloaded_model = self._load_volatility_model(DEFAULT_VOLATILITY_MODEL_PATH)
                if reloaded_model:
                    self.volatility_model = reloaded_model
                    is_sklearn_wrapper = hasattr(self.volatility_model, 'feature_names_in_')
                    is_native_xgboost = hasattr(self.volatility_model, 'get_booster') and hasattr(self.volatility_model.get_booster(), 'feature_names')
                    if not (is_sklearn_wrapper or (is_native_xgboost and self.volatility_model.get_booster().feature_names)):
                        logger.error("Reloaded default volatility model still appears untrained or lacks feature names. Cannot predict.")
                        return {"prediction": 0.0, "feature_names": []}
                else:
                    logger.error("Failed to reload default volatility model. Cannot predict.")
                    return {"prediction": 0.0, "feature_names": []}
            else:
                logger.error(f"Volatility model at {self.volatility_model_path} appears untrained or lacks feature names. Cannot predict.")
                return {"prediction": 0.0, "feature_names": []}

        if features_for_prediction.empty:
            logger.warning("Features DataFrame for volatility prediction is empty.")
            return {"prediction": 0.0, "feature_names": []}

        original_columns = features_for_prediction.columns.tolist()
        safe_feature_names_input = [col.replace('[', '_').replace(']', '_').replace('<', '_') for col in original_columns]
        
        features_df_safe_names = features_for_prediction.copy()
        features_df_safe_names.columns = safe_feature_names_input

        model_feature_names = []
        if is_native_xgboost and self.volatility_model.get_booster().feature_names:
            model_feature_names = self.volatility_model.get_booster().feature_names
        elif is_sklearn_wrapper and hasattr(self.volatility_model, 'feature_names_in_') and self.volatility_model.feature_names_in_ is not None:
            model_feature_names = self.volatility_model.feature_names_in_.tolist()
        
        if not model_feature_names:
            logger.error("Could not retrieve feature names from the trained model.")

        aligned_features_df = pd.DataFrame(columns=model_feature_names, index=features_df_safe_names.index)
        for col in model_feature_names:
            if col in features_df_safe_names.columns:
                aligned_features_df[col] = features_df_safe_names[col]
            else:
                logger.warning(f"Feature '{col}' expected by model not found in input. Prediction might be inaccurate.")
                aligned_features_df[col] = np.nan

        features_to_predict_with = aligned_features_df[model_feature_names]

        try:
            prediction = self.volatility_model.predict(features_to_predict_with)
            predicted_value = float(prediction[0]) if isinstance(prediction, (np.ndarray, list)) else float(prediction)
            
            return {"prediction": predicted_value, "feature_names": model_feature_names}

        except xgb.core.XGBoostError as xgb_err:
            logger.error(f"XGBoost prediction error: {xgb_err}. Model feature names: {model_feature_names}, Input features: {features_df_safe_names.columns.tolist()}")
            if "feature_names mismatch" in str(xgb_err):
                logger.error(
                    f"Feature names mismatch. Model expected: {model_feature_names}. "
                    f"Got after safe conversion: {features_df_safe_names.columns.tolist()}. "
                    f"Input before safe: {original_columns}"
                )
            return {"prediction": 0.0, "feature_names": model_feature_names if model_feature_names else []}
        except Exception as e:
            logger.exception(f"Error during volatility prediction: {e}")
            return {"prediction": 0.0, "feature_names": model_feature_names if model_feature_names else []}

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

    async def get_sentiment_insights(
        self, 
        texts_with_sentiment: List[Dict[str, Any]], 
        asset_symbol: str, 
        num_themes: int = 3
    ) -> Dict[str, Any]:
        """
        Analyzes a list of texts with their sentiment to distill key themes using an LLM.

        Args:
            texts_with_sentiment: A list of dictionaries, where each dict has:
                                  'text': The raw text (e.g., news headline, Reddit post title).
                                  'sentiment': The sentiment label (e.g., 'bullish', 'bearish', 'neutral').
                                  'score': The sentiment confidence score.
                                  'source': The source of the text (e.g., 'NewsAPI', 'Reddit').
            asset_symbol: The asset symbol these texts pertain to.
            num_themes: The desired number of key themes to extract.

        Returns:
            A dictionary with 'themes' (a list of strings) and 'summary' (a string),
            or an error message if the LLM is not available or an error occurs.
        """
        if not self.llm:
            logger.warning("LLM client not initialized. Cannot get sentiment insights.")
            return {"error": "LLM client not available. Please configure OPENAI_API_KEY."}

        if not texts_with_sentiment:
            return {"themes": [], "summary": "No texts provided to analyze."}

        # Prepare the content for the prompt
        formatted_texts = "\n".join(
            [
                f"- Source: {item.get('source', 'N/A')}, Sentiment: {item.get('sentiment', 'N/A')} ({item.get('score', 0.0):.2f}), Text: {item.get('text', '')}"
                for item in texts_with_sentiment
            ]
        )

        prompt_template = ChatPromptTemplate.from_messages(
            [
                ("system", "You are an expert financial analyst. Your task is to identify key themes from a list of texts related to a specific financial asset. Focus on the information that is most likely influencing market sentiment."),
                ("human", 
                 f"Asset: {asset_symbol}\n\n"
                 f"Recent news and discussions:\n{formatted_texts}\n\n"
                 f"Based on the texts above, identify the top {num_themes} key themes or news items that are currently driving sentiment for {asset_symbol}. "
                 f"For each theme, provide a concise label. Then, provide a brief overall summary of what these themes suggest about the current sentiment drivers for {asset_symbol}. "
                 f"Structure your response as follows:\n"
                 f"Themes:\n- [Theme 1]\n- [Theme 2]\n- ...\n"
                 f"Summary: [Your overall summary]")
            ]
        )
        
        chain = prompt_template | self.llm | StrOutputParser()

        try:
            logger.info(f"Requesting sentiment insights for {asset_symbol} from LLM.")
            llm_response_str = await chain.ainvoke({}) # Pass empty dict as input if prompt is self-contained
            
            # Basic parsing of the LLM response string
            themes = []
            summary = "Could not parse summary from LLM response." # Default
            
            lines = llm_response_str.split('\n')
            parsing_themes = False
            for line in lines:
                if line.strip().lower() == "themes:":
                    parsing_themes = True
                    continue
                if line.strip().lower().startswith("summary:"):
                    summary = line.split(":", 1)[1].strip()
                    parsing_themes = False # Stop parsing themes if summary is found
                    continue # Continue to next line in case summary is multi-line (though prompt asks for brief)
                
                if parsing_themes and line.strip().startswith("- "):
                    themes.append(line.strip()[2:])
            
            if not themes and "Themes:" not in llm_response_str: # Fallback if parsing fails
                 summary = llm_response_str # Put the whole response as summary if parsing fails

            logger.info(f"Received sentiment insights for {asset_symbol}: Themes: {themes}, Summary: {summary}")
            return {"themes": themes, "summary": summary}

        except Exception as e:
            logger.error(f"Error getting sentiment insights from LLM for {asset_symbol}: {e}")
            return {"error": f"LLM interaction failed: {str(e)}"}

    async def forecast_event_impact_score(
        self, 
        event_details: Dict[str, Any], 
    ) -> Dict[str, Any]:
        """
        Forecasts the potential market impact score of an economic event.
        MVP: Rule-based. Future: XGBoost classifier or similar.

        Args:
            event_details: A dictionary containing details of the event.
                           Expected keys: 'name' (str), 'impact' (str from source, e.g., 'High', 'Low'),
                           'currency' (str, e.g., 'USD'), 'country' (str, e.g., 'US').

        Returns:
            A dictionary with 'impact_score' (str: "Low", "Medium", "High") 
            and 'confidence' (float).
        """
        logger.info(f"Forecasting impact score for event: {event_details.get('name')}")

        source_impact = str(event_details.get('impact', 'low')).lower()
        event_name = str(event_details.get('name', '')).lower()
        currency = str(event_details.get('currency', '')).upper()

        predicted_score = "Low"
        confidence = 0.5 # Default confidence

        if "high" in source_impact:
            predicted_score = "High"
            confidence = 0.7
        elif "medium" in source_impact or "mod" in source_impact: # mod for moderate
            predicted_score = "Medium"
            confidence = 0.6
        elif "low" in source_impact:
            predicted_score = "Low"
            confidence = 0.6
        
        high_impact_keywords = ["interest rate decision", "gdp growth rate", "non-farm payrolls", "inflation rate", "cpi", "unemployment rate"]
        medium_impact_keywords = ["retail sales", "pmi", "consumer confidence", "trade balance", "industrial production"]

        for keyword in high_impact_keywords:
            if keyword in event_name:
                if predicted_score != "High": # Upgrade if not already high
                    predicted_score = "High"
                    confidence = max(confidence, 0.75)
                else: # If already high, slightly increase confidence
                    confidence = min(0.9, confidence + 0.1)
                break 
        
        if predicted_score != "High": # Only check medium if not already high
            for keyword in medium_impact_keywords:
                if keyword in event_name:
                    if predicted_score == "Low": # Upgrade Low to Medium
                        predicted_score = "Medium"
                        confidence = max(confidence, 0.65)
                    else: # If already medium, slightly increase confidence
                        confidence = min(0.8, confidence + 0.05)
                    break
        
        major_currencies = ["USD", "EUR", "JPY", "GBP", "CAD", "AUD", "CHF"]
        if currency in major_currencies and predicted_score == "Medium":
            confidence = min(0.85, confidence + 0.05)
        elif currency in major_currencies and predicted_score == "High":
            confidence = min(0.95, confidence + 0.05)

        confidence = round(max(0.3, min(0.95, confidence)), 2)

        logger.info(f"Predicted impact for {event_details.get('name')}: {predicted_score}, Confidence: {confidence}")
        return {"impact_score": predicted_score, "confidence": confidence}

    async def detect_anomalies_with_prophet(
        self, 
        data_df: pd.DataFrame,
        interval_width: float = 0.95,
        changepoint_prior_scale: float = 0.05
    ) -> pd.DataFrame:
        """
        Detects anomalies in a time series using Prophet.

        Args:
            data_df: Pandas DataFrame with 'ds' (datetime) and 'y' (numeric) columns.
            interval_width: Float, width of the uncertainty interval.
            changepoint_prior_scale: Float, flexibility of the trend changepoints.

        Returns:
            Pandas DataFrame with original data, Prophet forecast ('yhat', 'yhat_lower', 
            'yhat_upper'), and 'is_anomaly' (boolean) column.
            Returns an empty DataFrame if input is invalid or an error occurs.
        """
        if not isinstance(data_df, pd.DataFrame) or data_df.empty:
            logger.warning("Input data_df is empty or not a DataFrame for anomaly detection.")
            return pd.DataFrame()
        if 'ds' not in data_df.columns or 'y' not in data_df.columns:
            logger.warning("Input DataFrame must contain 'ds' and 'y' columns.")
            return pd.DataFrame()
        if data_df['ds'].isnull().any() or data_df['y'].isnull().any():
            logger.warning("Input DataFrame 'ds' or 'y' columns contain null values.")
            return pd.DataFrame()
        if len(data_df) < 2:
            logger.warning("Not enough data points (<2) for Prophet anomaly detection.")
            return data_df.assign(yhat=np.nan, yhat_lower=np.nan, yhat_upper=np.nan, is_anomaly=False)

        prophet_model = Prophet(interval_width=interval_width, changepoint_prior_scale=changepoint_prior_scale)
        
        try:
            logger.info(f"Fitting Prophet model for anomaly detection on {len(data_df)} data points.")
            prophet_model.fit(data_df[['ds', 'y']])
            forecast = prophet_model.predict(data_df[['ds']])
            
            result_df = pd.merge(data_df, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds', how='left')
            
            result_df['is_anomaly'] = (result_df['y'] < result_df['yhat_lower']) | (result_df['y'] > result_df['yhat_upper'])
            
            logger.info(f"Prophet anomaly detection complete. Found {result_df['is_anomaly'].sum()} anomalies.")
            return result_df

        except Exception as e:
            logger.exception(f"Error during Prophet anomaly detection: {e}")
            return data_df.assign(yhat=np.nan, yhat_lower=np.nan, yhat_upper=np.nan, is_anomaly=False)

    async def generate_ai_explanation(
        self, 
        query: str, 
        context_data: Dict[str, Any],
        system_prompt_override: Optional[str] = None
    ) -> str:
        """
        Generates an AI explanation using an LLM (e.g., GPT-4) based on a query and context data.

        Args:
            query: The user's question or the topic to explain.
            context_data: A dictionary containing relevant data to provide context for the explanation.
                         Example: {"asset_symbol": "AAPL", "recent_news": [...], "sentiment_score": 0.7}
            system_prompt_override: Optional system prompt to guide the LLM's persona and task.

        Returns:
            A string containing the AI-generated explanation.
            Returns an error message string if the LLM is not available or an error occurs.
        """
        if not self.llm:
            logger.warning("LLM client not initialized. Cannot generate AI explanation.")
            return "Error: AI Explanation feature is currently unavailable. Please ensure OPENAI_API_KEY is configured."

        # Prepare the context string for the prompt
        context_str = "\n".join([f"{key}: {value}" for key, value in context_data.items()])
        if not context_str:
            context_str = "No specific context data provided."

        system_message_content = system_prompt_override or (
            "You are a highly knowledgeable financial analyst AI. Your goal is to provide clear, concise, and insightful explanations. "
            "Base your explanation on the provided context and the user's query. Avoid making direct financial advice or predictions of future prices unless specifically asked to hypothesize based on data."
        )
        
        human_message_content = f"Context Data:\n{context_str}\n\nUser Query: {query}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_message_content),
            ("human", human_message_content)
        ])
        
        chain = prompt | self.llm | StrOutputParser()

        try:
            logger.info(f"Requesting AI explanation for query: '{query}' with context keys: {list(context_data.keys())}")
            llm_response_str = await chain.ainvoke({})
            logger.info(f"Received AI explanation for query: '{query}'")
            return llm_response_str
        except Exception as e:
            logger.error(f"Error generating AI explanation from LLM for query '{query}': {e}", exc_info=True)
            return f"Error: Could not generate AI explanation due to an internal error: {str(e)}"

    async def analyze_event_echo_with_prophet(
        self,
        historical_prices_df: pd.DataFrame, # Must have 'ds' (datetime) and 'y' (float, close price) columns, sorted by 'ds'
        past_event_dates: List[datetime],
        window_pre_event_days: int = 5, # Number of trading days before event day
        window_post_event_days: int = 10, # Number of trading days after event day
        min_events_for_pattern: int = 3
    ) -> Dict[str, Any]:
        """
        Analyzes historical price data around similar past events to find a typical "echo" pattern
        using normalized averaging and Prophet smoothing.

        Args:
            historical_prices_df: DataFrame with asset's historical prices ('ds', 'y').
                                  Must be sorted by date and have 'ds' as datetime objects.
            past_event_dates: List of datetime.datetime objects for past similar events.
            window_pre_event_days: Number of trading days to look before the event day (T-N).
            window_post_event_days: Number of trading days to look after the event day (T+N).
            min_events_for_pattern: Minimum number of past events required to compute a pattern.

        Returns:
            A dictionary containing:
                'echo_pattern': List of {'days_offset': int, 'avg_price_change_percentage': float}
                'events_analyzed': int (number of past event dates used in the analysis)
                'message': str (status message)
        """
        logger.info(f"Analyzing event echo with {len(past_event_dates)} past events. Window: T-{window_pre_event_days} to T+{window_post_event_days}")

        if not isinstance(historical_prices_df, pd.DataFrame) or historical_prices_df.empty:
            return {"echo_pattern": [], "events_analyzed": 0, "message": "Historical prices DataFrame is empty."}
        if 'ds' not in historical_prices_df.columns or 'y' not in historical_prices_df.columns:
            return {"echo_pattern": [], "events_analyzed": 0, "message": "Historical prices DataFrame must contain 'ds' and 'y' columns."}
        
        # Ensure 'ds' is datetime and DataFrame is sorted
        if not pd.api.types.is_datetime64_any_dtype(historical_prices_df['ds']):
            historical_prices_df['ds'] = pd.to_datetime(historical_prices_df['ds'])
        historical_prices_df = historical_prices_df.sort_values(by='ds').reset_index(drop=True)
        
        if len(past_event_dates) < min_events_for_pattern:
            return {"echo_pattern": [], "events_analyzed": 0, "message": f"Not enough past events ({len(past_event_dates)}) to analyze. Minimum required: {min_events_for_pattern}."}

        all_normalized_series = []
        processed_event_count = 0

        for event_date in past_event_dates:
            event_date_dt = pd.to_datetime(event_date.date()) # Ensure it's just the date part, as datetime
            
            # Find index of event_date or closest trading day before it
            event_day_iloc = historical_prices_df['ds'].searchsorted(event_date_dt, side='right') -1
            if event_day_iloc < 0 : # Event date is before any price data
                logger.warning(f"Event date {event_date_dt} is before the start of price data. Skipping.")
                continue
            
            # Define the window slice using iloc
            start_iloc = event_day_iloc - window_pre_event_days
            end_iloc = event_day_iloc + window_post_event_days + 1 # +1 because slice upper bound is exclusive

            if start_iloc < 0 or end_iloc > len(historical_prices_df):
                logger.warning(f"Data window for event {event_date_dt} is out of bounds of historical prices. Skipping.")
                continue

            event_window_df = historical_prices_df.iloc[start_iloc:end_iloc].copy()
            
            if len(event_window_df) != (window_pre_event_days + 1 + window_post_event_days):
                logger.warning(f"Event window for {event_date_dt} does not have the expected number of trading days. Has {len(event_window_df)}, expected {window_pre_event_days + 1 + window_post_event_days}. Skipping.")
                continue

            # Normalization base: price on T-1 (day before event day)
            # The event day is at index `window_pre_event_days` in `event_window_df`
            if window_pre_event_days == 0: # If pre_event_days is 0, T-1 is not in window. Use T0 as base.
                 base_price_iloc_in_window = 0 
            else:
                 base_price_iloc_in_window = window_pre_event_days -1 
            
            if base_price_iloc_in_window < 0 or base_price_iloc_in_window >= len(event_window_df):
                logger.warning(f"Base price for normalization for event {event_date_dt} is out of window bounds. Skipping.")
                continue
            
            base_price = event_window_df.iloc[base_price_iloc_in_window]['y']

            if base_price is None or base_price == 0:
                logger.warning(f"Base price for normalization for event {event_date_dt} is zero or None. Skipping.")
                continue
            
            event_window_df['normalized_y'] = ((event_window_df['y'] / base_price) - 1) * 100
            all_normalized_series.append(event_window_df['normalized_y'].values)
            processed_event_count += 1

        if processed_event_count < min_events_for_pattern:
            return {"echo_pattern": [], "events_analyzed": processed_event_count, "message": f"Not enough valid event windows ({processed_event_count}) to analyze after filtering. Minimum required: {min_events_for_pattern}."}

        # Aggregate the normalized series
        avg_normalized_series = np.mean(np.array(all_normalized_series), axis=0)
        
        # Create DataFrame for Prophet
        # Days offset from T-window_pre_event_days to T+window_post_event_days
        # Event day (T0) corresponds to days_offset = 0
        days_offsets = list(range(-window_pre_event_days, window_post_event_days + 1))
        
        prophet_df = pd.DataFrame({
            'ds': pd.to_datetime(days_offsets, unit='D', origin=pd.Timestamp('2000-01-01')), # Synthetic dates for Prophet
            'y': avg_normalized_series
        })

        try:
            # Use a fresh Prophet instance for this specific task
            echo_prophet_model = Prophet(daily_seasonality=False, weekly_seasonality=False, yearly_seasonality=False)
            # Add custom seasonality if there's a theoretical basis, or keep it simple.
            # For event echoes, a simple trend fit might be best.
            echo_prophet_model.fit(prophet_df)
            forecast = echo_prophet_model.predict(prophet_df[['ds']])
            
            echo_pattern = []
            for i, day_offset in enumerate(days_offsets):
                echo_pattern.append({
                    'days_offset': day_offset,
                    'avg_price_change_percentage': round(forecast.iloc[i]['yhat'], 2)
                })
            
            return {
                "echo_pattern": echo_pattern, 
                "events_analyzed": processed_event_count, 
                "message": f"Successfully analyzed {processed_event_count} past events."
            }
        except Exception as e:
            logger.error(f"Error fitting Prophet to aggregated event echo data: {e}", exc_info=True)
            # Fallback: return the raw averaged data without Prophet smoothing if Prophet fails
            raw_echo_pattern = [
                {'days_offset': days_offsets[i], 'avg_price_change_percentage': round(avg_normalized_series[i], 2)}
                for i in range(len(avg_normalized_series))
            ]
            return {
                "echo_pattern": raw_echo_pattern, 
                "events_analyzed": processed_event_count, 
                "message": f"Successfully analyzed {processed_event_count} past events (Prophet smoothing failed, returning raw average)."
            }

async def example_train_and_predict():
    factory = MLModelFactory()

    n_samples = 100
    data = {
        'historical_vol_20d': np.random.rand(n_samples) * 0.05 + 0.01,
        'sentiment_score': np.random.rand(n_samples) * 2 - 1,
        'vix_close': np.random.rand(n_samples) * 30 + 10
    }
    features_df = pd.DataFrame(data)
    
    target_series = (
        0.5 * features_df['historical_vol_20d'] + 
        0.3 * (features_df['sentiment_score'] * 0.01) + 
        0.2 * (features_df['vix_close'] / 1000) + 
        np.random.rand(n_samples) * 0.005
    )
    target_series = target_series.clip(0.001)

    training_success = await factory.train_volatility_model(features_df, target_series)
    
    if training_success:
        logger.info("Model trained successfully.")
        new_features_data = {
            'historical_vol_20d': [0.025],
            'sentiment_score': [0.5],
            'vix_close': [22.0]
        }
        new_features_df = pd.DataFrame(new_features_data)
        
        predicted_vol = await factory.predict_volatility(new_features_df)
        logger.info(f"Predicted volatility: {predicted_vol}")
    else:
        logger.error("Model training failed.")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    pass
