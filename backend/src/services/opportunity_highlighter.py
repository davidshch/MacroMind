"""
Service for identifying market opportunities based on volatility and sentiment.
"""
import logging
from typing import List, Dict, Optional
from datetime import datetime, date
import asyncio

from ..schemas.opportunity import MarketOpportunityHighlight, MarketOpportunityResponse
from ..services.volatility import VolatilityService
from ..services.sentiment_analysis import SentimentAnalysisService
from ..config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# For MVP, a predefined list of popular assets. This could come from a config or DB later.
POPULAR_ASSETS_MVP = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "NVDA", "BTC-USD", "ETH-USD"]

class OpportunityHighlightService:
    def __init__(
        self,
        volatility_service: VolatilityService,
        sentiment_service: SentimentAnalysisService
    ):
        self.volatility_service = volatility_service
        self.sentiment_service = sentiment_service
        # Thresholds can be made configurable later
        self.volatility_threshold = settings.OPPORTUNITY_VOLATILITY_THRESHOLD # Example: 0.02 (2% daily vol)
        self.strong_sentiment_threshold = settings.OPPORTUNITY_SENTIMENT_THRESHOLD # Example: 0.5

    async def get_market_opportunities(
        self, 
        asset_list: Optional[List[str]] = None
    ) -> MarketOpportunityResponse:
        """
        Identifies assets with high forecasted volatility and strong directional sentiment.

        Args:
            asset_list: Optional list of symbols to check. If None, uses POPULAR_ASSETS_MVP.

        Returns:
            MarketOpportunityResponse containing highlighted assets.
        """
        target_assets = asset_list if asset_list else POPULAR_ASSETS_MVP
        logger.info(f"Checking for market opportunities in assets: {target_assets}")

        highlighted_opportunities: List[MarketOpportunityHighlight] = []
        
        # Concurrently fetch data for all assets
        tasks = []
        for symbol in target_assets:
            # For volatility, we need a method that gives a simple forecast value.
            # Assuming VolatilityService has or will have a method like `get_latest_volatility_forecast`
            # or we adapt `calculate_and_predict_volatility` to simplify its output for this use case.
            # For MVP, let's assume `calculate_and_predict_volatility` returns a dict with `predicted_volatility`
            vol_task = self.volatility_service.calculate_and_predict_volatility(symbol)
            
            # For sentiment, `get_aggregated_sentiment` returns AggregatedSentimentResponse
            sent_task = self.sentiment_service.get_aggregated_sentiment(symbol, target_date=date.today())
            tasks.append((symbol, vol_task, sent_task))

        results = []
        for symbol, vol_task, sent_task in tasks:
            try:
                vol_data, sent_data = await asyncio.gather(vol_task, sent_task, return_exceptions=True)
                results.append((symbol, vol_data, sent_data))
            except Exception as e:
                logger.error(f"Error gathering data for opportunity check on {symbol}: {e}")
                results.append((symbol, e, None)) # Mark error for this symbol
        
        for symbol, vol_data, sent_data in results:
            if isinstance(vol_data, Exception):
                logger.warning(f"Could not retrieve volatility data for {symbol} for opportunity check: {vol_data}")
                continue
            if isinstance(sent_data, Exception):
                logger.warning(f"Could not retrieve sentiment data for {symbol} for opportunity check: {sent_data}")
                continue
            if not vol_data or not sent_data:
                logger.debug(f"Skipping {symbol} due to missing volatility or sentiment data.")
                continue

            # Extract relevant values - adjust keys based on actual service responses
            # From VolatilityService.calculate_and_predict_volatility response
            predicted_vol = vol_data.get('predicted_volatility') 
            current_vol = vol_data.get('current_volatility')

            # From SentimentAnalysisService.get_aggregated_sentiment response (AggregatedSentimentResponse schema)
            sentiment_score = sent_data.normalized_score
            sentiment_label = str(sent_data.overall_sentiment.value) # Get string value of Enum

            if predicted_vol is None or sentiment_score is None:
                logger.debug(f"Missing predicted volatility or sentiment score for {symbol}.")
                continue

            is_high_volatility = predicted_vol > self.volatility_threshold
            is_strong_bullish = sentiment_score > self.strong_sentiment_threshold
            is_strong_bearish = sentiment_score < -self.strong_sentiment_threshold

            reason = ""
            if is_high_volatility and is_strong_bullish:
                reason = f"High Forecasted Volatility ({predicted_vol:.4f}) & Strong Bullish Sentiment ({sentiment_score:.2f})"
            elif is_high_volatility and is_strong_bearish:
                reason = f"High Forecasted Volatility ({predicted_vol:.4f}) & Strong Bearish Sentiment ({sentiment_score:.2f})"
            
            if reason:
                logger.info(f"Market opportunity highlighted for {symbol}: {reason}")
                highlighted_opportunities.append(
                    MarketOpportunityHighlight(
                        symbol=symbol,
                        timestamp=datetime.now(),
                        reason=reason,
                        # For MVP, we might not need the full VolatilityForecast schema here
                        # vol_forecast_details can be added if schema is compatible or adapted
                        current_volatility=current_vol,
                        sentiment_score=sentiment_score,
                        sentiment_label=sentiment_label
                    )
                )

        return MarketOpportunityResponse(
            generated_at=datetime.now(),
            highlights=highlighted_opportunities,
            assets_checked=len(target_assets),
            message=f"Found {len(highlighted_opportunities)} potential opportunities out of {len(target_assets)} assets checked."
        )
