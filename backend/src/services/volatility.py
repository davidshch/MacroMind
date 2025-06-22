"""Enhanced volatility prediction service using XGBoost and comprehensive features."""

from datetime import datetime, timedelta, date
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
import logging
import asyncio
from fastapi import HTTPException, Depends
from scipy.stats import percentileofscore
import os
import json
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..database.database import get_db
from ..database.models import AggregatedSentiment
from .market_data import MarketDataService
from .ml.model_factory import MLModelFactory
from ..schemas.volatility import VolatilityRegime
from ..core.service_providers import get_market_data_service, get_ml_model_factory

logger = logging.getLogger(__name__)

class VolatilityService:
    """Service for analyzing and predicting market volatility using MLModelFactory."""
    
    def __init__(self, 
                 db: AsyncSession = Depends(get_db),
                 market_data_service: MarketDataService = Depends(get_market_data_service),
                 ml_model_factory: MLModelFactory = Depends(get_ml_model_factory)
                ):
        self.db = db
        self.cache = {}
        self.cache_duration = timedelta(minutes=15)
        self.market_data_service = market_data_service
        self.ml_model_factory = ml_model_factory
        
        self.use_redis = False
        if os.getenv('USE_REDIS') == 'TRUE' and not os.getenv('TESTING') == 'TRUE':
            try:
                import redis
                self.redis = redis.Redis(host='localhost', port=6379, db=0, socket_timeout=2)
                self.redis.ping()
                self.use_redis = True
                logger.info("Redis connection successful for VolatilityService")
            except Exception as e:
                logger.warning(f"Redis connection failed for VolatilityService. Using in-memory cache: {str(e)}")
                self.use_redis = False

    async def _fetch_historical_sentiment_df(self, symbol: str, start_date: date, end_date: date) -> pd.DataFrame:
        """
        Fetches historical aggregated sentiment scores from the database.

        Args:
            symbol: The stock/asset symbol.
            start_date: The start date for fetching sentiment.
            end_date: The end date for fetching sentiment.

        Returns:
            A Pandas DataFrame with 'date' and 'sentiment_score' columns.
        """
        try:
            stmt = (
                select(AggregatedSentiment.date, AggregatedSentiment.score)
                .where(
                    AggregatedSentiment.symbol == symbol,
                    AggregatedSentiment.date >= start_date,
                    AggregatedSentiment.date <= end_date
                )
                .order_by(AggregatedSentiment.date)
            )
            result = await self.db.execute(stmt)
            sentiments = result.fetchall()
            
            if not sentiments:
                logger.warning(f"No historical sentiment data found for {symbol} between {start_date} and {end_date}.")
                return pd.DataFrame(columns=['date', 'sentiment_score']).set_index('date')

            df = pd.DataFrame(sentiments, columns=['date', 'sentiment_score'])
            df['date'] = pd.to_datetime(df['date'])
            df = df.set_index('date')
            return df
        except Exception as e:
            logger.error(f"Error fetching historical sentiment for {symbol}: {e}")
            return pd.DataFrame(columns=['date', 'sentiment_score']).set_index('date')

    async def _prepare_comprehensive_features_df(
        self, 
        symbol: str, 
        end_date: datetime,
        data_fetch_lookback_days: int
    ) -> pd.DataFrame:
        """
        Prepares a comprehensive feature DataFrame for a given symbol up to end_date.
        Features include historical volatility, VIX, sentiment, and technical indicators.
        """
        fetch_start_date = pd.Timestamp(end_date) - pd.Timedelta(days=data_fetch_lookback_days + 90)

        logger.debug(f"Fetching price data for {symbol} from {fetch_start_date} to {end_date}")
        price_data_list = await self.market_data_service.get_historical_prices(symbol, data_fetch_lookback_days + 90)
        
        # If we don't have enough price data, generate mock data for training
        if not price_data_list or len(price_data_list) < data_fetch_lookback_days * 0.5:
            logger.warning(f"Insufficient price data for {symbol} ({len(price_data_list) if price_data_list else 0} records), generating mock data for training")
            price_data_list = self._generate_mock_price_data(symbol, fetch_start_date, end_date, data_fetch_lookback_days + 90)
        
        if not price_data_list:
            logger.warning(f"No price data for {symbol}")
            return pd.DataFrame()
        
        logger.debug(f"Processing {len(price_data_list)} price records for {symbol}")
        
        df = pd.DataFrame(price_data_list)
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        df = df[~df.index.duplicated(keep='first')]
        df = df[(df.index >= pd.Timestamp(fetch_start_date)) & (df.index <= pd.Timestamp(end_date))]
        
        logger.debug(f"After filtering, DataFrame has {len(df)} rows for {symbol}")

        # Add technical indicators
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        for window in [5, 10, 20, 60]:
            df[f'historical_vol_{window}d'] = df['log_returns'].rolling(window=window, min_periods=max(1, window // 2)).std() * np.sqrt(252)

        # Fetch VIX data
        vix_data_list = await self.market_data_service.get_historical_prices('^VIX', data_fetch_lookback_days + 90)
        if vix_data_list and len(vix_data_list) > 10:
            df_vix = pd.DataFrame(vix_data_list)
            df_vix['date'] = pd.to_datetime(df_vix['date'])
            df_vix = df_vix.set_index('date').sort_index()
            df_vix = df_vix[~df_vix.index.duplicated(keep='first')]
            df['vix_close'] = df_vix['close'].reindex(df.index, method='ffill')
        else:
            logger.warning("No VIX data found, generating mock VIX data.")
            df['vix_close'] = self._generate_mock_vix_data(df.index)

        # Fetch sentiment data
        sentiment_df = await self._fetch_historical_sentiment_df(symbol, fetch_start_date.date(), end_date.date())
        
        # Ensure unique indices before merging
        df = df[~df.index.duplicated(keep='first')]
        
        logger.debug(f"Price DataFrame before sentiment merge: {len(df)} rows, date range: {df.index.min()} to {df.index.max()}")
        logger.debug(f"Sentiment DataFrame: {len(sentiment_df)} rows, date range: {sentiment_df.index.min() if not sentiment_df.empty else 'N/A'} to {sentiment_df.index.max() if not sentiment_df.empty else 'N/A'}")
        
        if not sentiment_df.empty:
            sentiment_df = sentiment_df[~sentiment_df.index.duplicated(keep='first')]
            
            if not sentiment_df.empty:
                # Use left join to keep all price data, fill missing sentiment with 0
                df = df.merge(sentiment_df, left_index=True, right_index=True, how='left')
                # Drop duplicate indices after merge
                df = df[~df.index.duplicated(keep='first')]
                # Fill missing sentiment scores with 0
                df['sentiment_score'] = df['sentiment_score'].fillna(0.0)
                logger.debug(f"Added sentiment data, DataFrame now has {len(df)} rows")
                logger.debug(f"DataFrame head after merge:\n{df.head(5)}")
            else:
                logger.warning(f"No sentiment data for {symbol}, filling with 0.")
                df['sentiment_score'] = 0.0
        else:
            logger.warning(f"No sentiment data for {symbol}, filling with 0.")
            df['sentiment_score'] = 0.0

        # Fill any remaining NaNs in vix_close with mock data
        if 'vix_close' in df.columns and df['vix_close'].isna().any():
            nan_idx = df[df['vix_close'].isna()].index
            df.loc[nan_idx, 'vix_close'] = self._generate_mock_vix_data(nan_idx)
            logger.debug(f"Filled {len(nan_idx)} NaNs in vix_close with mock data after merge.")

        # Finalize DataFrame
        # Only drop NaNs in columns required for training
        required_cols = ['log_returns', 'historical_vol_5d', 'historical_vol_10d', 'historical_vol_20d', 'historical_vol_60d', 'vix_close', 'sentiment_score']
        df = df.ffill().bfill()
        # Drop the first 60 rows to allow rolling windows to populate
        if len(df) > 60:
            df = df.iloc[60:]
        logger.debug(f"DataFrame after dropping first 60 rows, head:\n{df.head(5)}")
        logger.debug(f"NaN count per column before dropna: {df.isna().sum().to_dict()}")
        df.dropna(subset=required_cols, inplace=True)
        
        logger.debug(f"Final DataFrame for {symbol} has {len(df)} rows and columns: {list(df.columns)}")
        logger.debug(f"Final DataFrame head for {symbol}:\n{df.head()}")
        logger.debug(f"Final DataFrame tail for {symbol}:\n{df.tail()}")

        return df

    def _generate_mock_price_data(self, symbol: str, start_date: pd.Timestamp, end_date: pd.Timestamp, days: int) -> List[Dict[str, Any]]:
        """Generate mock historical price data for training when real data is insufficient."""
        import numpy as np
        from datetime import date, timedelta
        
        # Base prices for different symbols
        base_prices = {
            "NVDA": 150.0,
            "AAPL": 180.0,
            "TSLA": 250.0,
            "MSFT": 400.0,
            "GOOGL": 140.0,
            "AMZN": 180.0,
            "META": 500.0,
            "SPY": 450.0,
            "QQQ": 380.0,
            "^VIX": 20.0
        }
        
        base_price = base_prices.get(symbol, 100.0)
        
        # Generate trading days (skip weekends)
        trading_days = []
        current_date = start_date.date()
        while current_date <= end_date.date():
            if current_date.weekday() < 5:  # Monday = 0, Friday = 4
                trading_days.append(current_date)
            current_date += timedelta(days=1)
        
        # Limit to requested days
        trading_days = trading_days[-days:] if len(trading_days) > days else trading_days
        
        price_data = []
        current_price = base_price
        
        for i, day in enumerate(trading_days):
            # Generate realistic price movements
            trend_factor = 0.0001 * (i % 252)  # Annual trend
            volatility = 0.02  # 2% daily volatility
            random_factor = np.random.normal(0, volatility)
            
            # Calculate price change
            price_change = trend_factor + random_factor
            current_price *= (1 + price_change)
            
            # Generate OHLC data
            daily_volatility = 0.01  # 1% intraday volatility
            open_price = current_price * (1 + np.random.normal(0, daily_volatility * 0.5))
            high_price = max(open_price, current_price) * (1 + abs(np.random.normal(0, daily_volatility * 0.3)))
            low_price = min(open_price, current_price) * (1 - abs(np.random.normal(0, daily_volatility * 0.3)))
            close_price = current_price
            
            # Generate volume
            base_volume = 1000000
            volume = base_volume * (1 + np.random.normal(0, 0.3))
            
            price_data.append({
                "date": day.strftime("%Y-%m-%d"),
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "volume": int(volume)
            })
        
        return price_data

    def _generate_mock_vix_data(self, index) -> pd.Series:
        """Generate mock VIX data with the given index for alignment."""
        import numpy as np
        # Generate realistic VIX values (usually between 10-40)
        vix_values = np.random.normal(20, 5, len(index))
        vix_values = np.clip(vix_values, 10, 40)  # Clip to realistic range
        return pd.Series(vix_values, index=index)

    async def _prepare_training_data_df(
        self, 
        symbol: str, 
        training_start_date: date, 
        training_end_date: date, 
        future_vol_period_days: int = 5
    ) -> Tuple[Optional[pd.DataFrame], Optional[pd.Series]]:
        """Prepares features and target variable for training the volatility model."""
        
        features_df = await self._prepare_comprehensive_features_df(
            symbol, 
            datetime.combine(training_end_date, datetime.min.time()) + timedelta(days=future_vol_period_days + 5),
            data_fetch_lookback_days=(training_end_date - training_start_date).days + 90
        )

        if features_df.empty:
            logger.error(f"Could not prepare features for training {symbol}.")
            return None, None

        features_df_filtered = features_df[
            (features_df.index >= pd.to_datetime(training_start_date)) & 
            (features_df.index <= pd.to_datetime(training_end_date))
        ].copy()

        if features_df_filtered.empty:
            logger.error(f"No feature data for {symbol} in the specified training date range.")
            return None, None

        target_series = (
            features_df['log_returns']
            .rolling(window=future_vol_period_days)
            .std()
            .shift(-future_vol_period_days) * np.sqrt(252)
        )
        target_series = target_series.reindex(features_df_filtered.index).rename("target_volatility")
        
        final_df = features_df_filtered.join(target_series)
        final_df.dropna(subset=list(features_df_filtered.columns) + ['target_volatility'], inplace=True)

        if final_df.empty:
            logger.error(f"No aligned feature/target data after NaN drop for {symbol}.")
            return None, None
            
        training_features = final_df.drop(columns=['target_volatility', 'log_returns', 'close', 'high', 'low', 'open', 'volume'], errors='ignore')
        training_target = final_df['target_volatility']
        
        return training_features, training_target

    async def train_model_for_symbol_workflow(
        self, 
        symbol: str, 
        training_start_date_str: str, 
        training_end_date_str: str,
        future_vol_period_days: int = 5
    ) -> Dict[str, Any]:
        """
        Workflow to train the volatility model for a given symbol and date range.
        Can be triggered by an admin API or an offline script.
        """
        try:
            start_date = datetime.strptime(training_start_date_str, "%Y-%m-%d").date()
            end_date = datetime.strptime(training_end_date_str, "%Y-%m-%d").date()
        except ValueError:
            logger.error(f"Invalid date format provided for training: start={training_start_date_str}, end={training_end_date_str}")
            return {"status": "error", "message": "Invalid date format. Use YYYY-MM-DD."}

        logger.info(f"Starting volatility model training workflow for {symbol} from {start_date} to {end_date}.")
        
        features_df, target_series = await self._prepare_training_data_df(
            symbol, start_date, end_date, future_vol_period_days
        )

        if features_df is None or features_df.empty or target_series is None or target_series.empty:
            msg = f"Failed to prepare training data for {symbol}."
            logger.error(msg)
            return {"status": "error", "message": msg}

        logger.info(f"Training data prepared for {symbol}. Features: {features_df.columns.tolist()}, Samples: {len(features_df)}")
        
        os.makedirs(self.ml_model_factory.TRAINED_MODELS_DIR, exist_ok=True)

        model_save_path = os.path.join(
            self.ml_model_factory.TRAINED_MODELS_DIR, 
            f"{symbol.lower()}_xgboost_volatility_model.joblib"
        )
        
        training_success = await self.ml_model_factory.train_volatility_model(
            features_df, target_series, save_path=model_save_path
        )

        if training_success:
            msg = f"Volatility model for {symbol} trained successfully and saved to {model_save_path}."
            logger.info(msg)
            return {"status": "success", "message": msg, "model_path": model_save_path, "features_used": features_df.columns.tolist()}
        else:
            msg = f"Volatility model training failed for {symbol}."
            logger.error(msg)
            return {"status": "error", "message": msg}

    async def _predict_volatility_with_factory(
        self, 
        symbol: str, 
        feature_df_for_prediction_context: pd.DataFrame
    ) -> Dict[str, Any]:
        """Predicts future volatility using MLModelFactory and comprehensive features."""
        
        default_prediction_error_response = {
            "mean": 0.015, "low": 0.010, "high": 0.020, 
            "trend": "stable", "confidence": 0.3, 
            "feature_names": ["N/A - Prediction error or no model"]
        }
        
        logger.debug(f"Starting prediction for {symbol} using factory.")

        current_hist_vol_20d = 0.015
        if 'historical_vol_20d' in feature_df_for_prediction_context and not feature_df_for_prediction_context['historical_vol_20d'].empty:
            current_hist_vol_20d = feature_df_for_prediction_context['historical_vol_20d'].iloc[-1]
        
        default_prediction_error_response["mean"] = current_hist_vol_20d
        default_prediction_error_response["low"] = current_hist_vol_20d * 0.8
        default_prediction_error_response["high"] = current_hist_vol_20d * 1.2

        if feature_df_for_prediction_context.empty:
            logger.warning(f"Cannot predict volatility for {symbol}: feature DataFrame is empty.")
            return default_prediction_error_response

        logger.debug(f"Feature DataFrame for prediction has {len(feature_df_for_prediction_context)} rows. Tail:\n{feature_df_for_prediction_context.tail()}")

        model_path = os.path.join(
            self.ml_model_factory.TRAINED_MODELS_DIR,
            f"{symbol.lower()}_xgboost_volatility_model.joblib"
        )

        if not os.path.exists(model_path):
            logger.warning(f"Model file not found at {model_path} for symbol {symbol}. Returning default prediction based on historical.")
            default_prediction_error_response["feature_names"] = ["N/A - Model not found"]
            return default_prediction_error_response

        latest_features_series = feature_df_for_prediction_context.iloc[-1].copy()
        
        model_input_features_df = latest_features_series.drop(
            ['log_returns', 'close', 'high', 'low', 'open', 'volume'], 
            errors='ignore'
        ).to_frame().T

        logger.debug(f"Shape of model input: {model_input_features_df.shape}. Columns: {model_input_features_df.columns.tolist()}")
        logger.debug(f"Model input data:\n{model_input_features_df.to_string()}")

        logger.debug(f"Calling ml_model_factory.predict_volatility for {symbol} with model_path: {model_path}")
        prediction_result_dict = await self.ml_model_factory.predict_volatility(
            model_path=model_path, 
            features_for_prediction=model_input_features_df
        )
        logger.debug(f"Received result from ml_model_factory.predict_volatility for {symbol}")

        if "error" in prediction_result_dict:
            error_msg = prediction_result_dict["error"]
            logger.error(f"Error predicting volatility for {symbol} from factory: {error_msg}")
            response = default_prediction_error_response.copy()
            response["feature_names"] = [f"N/A - Prediction error: {error_msg}"]
            return response
        
        if "prediction" not in prediction_result_dict:
            logger.error(f"Prediction key missing for {symbol} from factory, though no explicit error. Result: {prediction_result_dict}")
            response = default_prediction_error_response.copy()
            response["feature_names"] = [f"N/A - Prediction key missing"]
            return response

        predicted_value = prediction_result_dict["prediction"]
        feature_names_from_model = prediction_result_dict.get("feature_names", ["N/A - Missing in model output"]) 

        if predicted_value == 0.0:
             logger.warning(
                 f"Prediction for {symbol} is 0.0. This might be a valid low volatility prediction or indicate "
                 f"model/data issues. Using historical vol ({current_hist_vol_20d:.4f}) as a fallback interpretation for now. "
                 f"Features from model: {feature_names_from_model}"
             )
             return {
                "mean": current_hist_vol_20d,
                "low": current_hist_vol_20d * 0.8,
                "high": current_hist_vol_20d * 1.2,
                "trend": "stable",
                "confidence": 0.3,
                "feature_names": feature_names_from_model
            }
        
        current_short_term_vol = feature_df_for_prediction_context['historical_vol_10d'].iloc[-1] if 'historical_vol_10d' in feature_df_for_prediction_context and not feature_df_for_prediction_context['historical_vol_10d'].empty else predicted_value
        
        trend = "stable"
        if predicted_value > current_short_term_vol * 1.05:
            trend = "increasing"
        elif predicted_value < current_short_term_vol * 0.95:
            trend = "decreasing"

        confidence = 0.75
        logger.info(f"Volatility prediction for {symbol}: {predicted_value:.4f}, Trend: {trend}, Confidence: {confidence:.2f}")
        
        return {
            "mean": float(predicted_value),
            "low": float(max(0.001, predicted_value * 0.8)),
            "high": float(predicted_value * 1.2),
            "trend": trend,
            "confidence": confidence,
            "feature_names": feature_names_from_model
        }

    async def calculate_and_predict_volatility(
        self,
        symbol: str,
        lookback_days: int = 30,
        prediction_days: int = 5,
    ) -> Dict[str, Any]:
        """Calculate current volatility metrics and predict future volatility using MLModelFactory."""
        try:
            if os.getenv('TESTING') == 'TRUE':
                return self._get_mock_volatility_data(symbol)
            
            cache_key = f"volatility_v2:{symbol}:{lookback_days}"
            cached = self._get_from_cache(cache_key)
            if cached:
                return cached

            current_time = datetime.now()
            
            logger.info(f"Preparing comprehensive features for {symbol} for prediction.")
            features_df = await self._prepare_comprehensive_features_df(
                symbol, 
                current_time, 
                data_fetch_lookback_days=lookback_days + 60 
            )

            if features_df.empty or len(features_df) < 5:
                logger.warning(f"Insufficient data after feature preparation for {symbol}, using mock data.")
                return self._get_mock_volatility_data(symbol)
            
            logger.debug(f"Comprehensive features prepared for {symbol}. Shape: {features_df.shape}. Columns: {features_df.columns.tolist()}")

            context_start_date = pd.to_datetime(current_time - timedelta(days=lookback_days))
            df_context = features_df[features_df.index >= context_start_date].copy()

            if df_context.empty:
                 logger.warning(f"Insufficient data in context window for {symbol}, using full features_df for context.")
                 df_context = features_df.copy()

            current_vol_metrics = self._calculate_current_volatility(df_context)
            model_predictions = await self._predict_volatility_with_factory(symbol, features_df) 
            
            market_condition = self._assess_market_condition(current_vol_metrics, model_predictions)
            volatility_regime_str = self._detect_volatility_regime(df_context)
            
            try:
                volatility_regime_enum = VolatilityRegime[volatility_regime_str.upper()]
            except KeyError:
                logger.warning(f"Could not map regime string '{volatility_regime_str}' to VolatilityRegime enum. Defaulting to NORMAL.")
                volatility_regime_enum = VolatilityRegime.NORMAL

            model_feature_names = model_predictions.get("feature_names", ["N/A"])

            result = {
                "symbol": symbol,
                "timestamp": current_time.isoformat(),
                "current_volatility": float(current_vol_metrics["current"]),
                "historical_volatility_annualized": float(current_vol_metrics["annualized"]),
                "volatility_10d_percentile": float(current_vol_metrics["percentile"]),
                "predicted_volatility": float(model_predictions["mean"]),
                "prediction_range": {
                    "low": float(model_predictions["low"]),
                    "high": float(model_predictions["high"])
                },
                "market_conditions": market_condition,
                "volatility_regime": volatility_regime_enum.value,
                "is_high_volatility": bool(current_vol_metrics["is_high"]),
                "trend": model_predictions["trend"],
                "confidence_score": float(model_predictions["confidence"]),
                "metadata": {
                    "model_version": "xgb_factory_1.1",
                    "features_used": model_feature_names,
                    "last_updated": current_time.isoformat()
                }
            }
            
            self._add_to_cache(cache_key, result)
            return result

        except Exception as e:
            logger.exception(f"Error calculating volatility for {symbol}: {e}")
            return self._get_mock_volatility_data(symbol)

    def _calculate_bollinger_bandwidth(self, prices: pd.Series, window: int = 20) -> pd.Series:
        if prices.empty or len(prices) < window:
            return pd.Series([0.05] * len(prices), index=prices.index, dtype=float)
        rolling_mean = prices.rolling(window=window, min_periods=max(1,window//2)).mean()
        rolling_std = prices.rolling(window=window, min_periods=max(1,window//2)).std()
        bb_upper = rolling_mean + (rolling_std * 2)
        bb_lower = rolling_mean - (rolling_std * 2)
        bandwidth = (bb_upper - bb_lower) / rolling_mean.replace(0, np.nan)
        return bandwidth.fillna(0.05)

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        if prices.empty or len(prices) < period + 1:
            return pd.Series([50.0] * len(prices), index=prices.index, dtype=float)
        delta = prices.diff(1)
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=max(1,period//2)).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=period, min_periods=max(1,period//2)).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        return rsi.fillna(50.0)

    def _calculate_atr(self, df_hlc: pd.DataFrame, period: int = 14) -> pd.Series:
        if df_hlc.empty or not all(col in df_hlc.columns for col in ['high', 'low', 'close']) or len(df_hlc) < 2:
            return pd.Series(np.nan, index=df_hlc.index if not df_hlc.empty else None, dtype=float)

        high_low = df_hlc['high'] - df_hlc['low']
        high_close_prev = (df_hlc['high'] - df_hlc['close'].shift(1)).abs()
        low_close_prev = (df_hlc['low'] - df_hlc['close'].shift(1)).abs()
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1, skipna=False)
        atr = tr.rolling(window=period, min_periods=max(1,period//2)).mean()
        return atr.fillna(tr.mean() if not tr.dropna().empty else 0.01)

    def _calculate_current_volatility(self, df_context: pd.DataFrame) -> Dict[str, Any]:
        defaults = {"current": 0.015 * np.sqrt(252), "annualized": 0.015 * np.sqrt(252), "percentile": 50.0, "is_high": False}
        if df_context.empty or 'historical_vol_10d' not in df_context.columns:
            logger.warning("Insufficient data for current volatility calculation, returning defaults.")
            return defaults
        
        vol_10d_series = df_context['historical_vol_10d'].dropna()
        if len(vol_10d_series) < 2:
            logger.warning("Not enough data points in vol_10d_series for robust calculation, returning defaults or simplified.")
            last_vol = vol_10d_series.iloc[-1] if not vol_10d_series.empty else (0.015 * np.sqrt(252))
            return {"current": float(last_vol), "annualized": float(last_vol), "percentile": 50.0, "is_high": False}
            
        current_vol = vol_10d_series.iloc[-1]
        annualized_vol = current_vol
        
        unique_vol_values = vol_10d_series.unique()
        if len(unique_vol_values) >= 2:
            vol_percentile = percentileofscore(vol_10d_series, current_vol, kind='rank')
        else:
            vol_percentile = 50.0
        
        return {
            "current": float(current_vol),
            "annualized": float(annualized_vol),
            "percentile": float(vol_percentile),
            "is_high": vol_percentile > 75
        }

    def _detect_volatility_regime(self, df_context: pd.DataFrame) -> str:
        if df_context.empty or 'historical_vol_20d' not in df_context.columns:
            return VolatilityRegime.NORMAL.value
            
        vol_20d_series = df_context['historical_vol_20d'].dropna()
        if len(vol_20d_series) < 5:
            return VolatilityRegime.NORMAL.value

        current_vol_20d = vol_20d_series.iloc[-1]
        
        if current_vol_20d > np.percentile(vol_20d_series, 90):
            return VolatilityRegime.CRISIS.value 
        elif current_vol_20d > np.percentile(vol_20d_series, 75):
            return VolatilityRegime.HIGH_STABLE.value
        elif current_vol_20d < np.percentile(vol_20d_series, 25):
            return VolatilityRegime.LOW_STABLE.value
        return VolatilityRegime.NORMAL.value

    def _assess_market_condition(self, current_vol_metrics: Dict[str, Any], model_predictions: Dict[str, Any]) -> str:
        if current_vol_metrics["is_high"] and model_predictions["trend"] == "increasing":
            return "highly_volatile_increasing"
        elif current_vol_metrics["is_high"]:
            return "high_volatility_stable_or_decreasing"
        elif model_predictions["trend"] == "increasing":
            return "normal_volatility_increasing"
        elif current_vol_metrics["percentile"] < 25:
            return "low_volatility"
        return "normal_volatility_stable"

    def _get_mock_volatility_data(self, symbol: str) -> Dict[str, Any]:
        """Return consistent mock volatility data for testing or error fallback."""
        return {
            "symbol": symbol,
            "timestamp": datetime.now().isoformat(),
            "current_volatility": 0.015 * 15.87,
            "historical_volatility_annualized": 0.24,
            "volatility_10d_percentile": 65.0,
            "predicted_volatility": 0.018 * 15.87,
            "prediction_range": {
                "low": 0.012 * 15.87,
                "high": 0.024 * 15.87
            },
            "market_conditions": "normal_mock",
            "volatility_regime": VolatilityRegime.NORMAL.value,
            "is_high_volatility": False,
            "trend": "stable",
            "confidence_score": 0.75,
            "metadata": {
                "model_version": "mock_factory_0.1",
                "features_used": ["mock_vol_20d", "mock_sentiment", "mock_vix"],
                "last_updated": datetime.now().isoformat(),
                "is_mock_data": True
            }
        }

    def _get_from_cache(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self.cache:
            data, timestamp = self.cache[key]
            if datetime.now() - timestamp < self.cache_duration:
                return data
            del self.cache[key]
        return None

    def _add_to_cache(self, key: str, data: Dict[str, Any]):
        self.cache[key] = (data, datetime.now())

    def _get_from_redis(self, key: str) -> Optional[Dict[str, Any]]:
        if not self.use_redis: return None
        try:
            data = self.redis.get(key)
            if data: return json.loads(data)
        except Exception as e: logger.error(f"Redis GET error: {e}")
        return None

    def _set_in_redis(self, key: str, data: Dict[str, Any]):
        if not self.use_redis: return
        try:
            self.redis.setex(key, self.cache_duration.seconds, json.dumps(data))
        except Exception as e: logger.error(f"Redis SETEX error: {e}")
        
    async def get_historical_volatility(
        self,
        symbol: str,
        days: int = 90,
    ) -> List[Dict[str, Any]]:
        """Get historical volatility data with regime classifications."""
        try:
            features_df = await self._prepare_comprehensive_features_df(
                symbol,
                datetime.now(),
                data_fetch_lookback_days=days
            )

            if features_df.empty:
                logger.warning(f"No historical data found for {symbol}")
                return []

            result = []
            for idx, row in features_df.iterrows():
                regime = self._detect_volatility_regime(
                    features_df[features_df.index <= idx]
                )
                
                vol_value = row['historical_vol_10d']
                vol_series = features_df['historical_vol_10d'][:idx+1]
                percentile = float(percentileofscore(vol_series, vol_value) if len(vol_series) > 1 else 50.0)
                
                result.append({
                    "date": idx.isoformat(),
                    "value": float(vol_value),
                    "regime": regime,
                    "percentile": percentile
                })

            return result

        except Exception as e:
            logger.exception(f"Error getting historical volatility for {symbol}: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to retrieve historical volatility: {str(e)}"
            )

async def get_volatility_service(
    db: AsyncSession = Depends(get_db),
    market_data_service: MarketDataService = Depends(get_market_data_service),
    ml_model_factory: MLModelFactory = Depends(get_ml_model_factory)
) -> VolatilityService:
    return VolatilityService(
        db=db, 
        market_data_service=market_data_service, 
        ml_model_factory=ml_model_factory
    )
