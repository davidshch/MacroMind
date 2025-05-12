"""AI-powered market analysis and explanation service."""

import logging
from typing import Dict, Any, Optional, List
from datetime import datetime, date, timedelta
import asyncio
import json
from fastapi import HTTPException, Depends

from ..config import get_settings
from .market_data import MarketDataService # Keep for type hint
from .sentiment_analysis import SentimentAnalysisService # Keep for type hint
from .volatility import VolatilityService # Keep for type hint
from .economic_calendar import EconomicCalendarService, get_economic_calendar_service # Keep for type hint and its own provider
from .ml.model_factory import MLModelFactory, get_ml_model_factory

# Import new centralized providers
from ..core.dependencies import (
    get_market_data_service, 
    get_sentiment_service, 
    get_volatility_service
)

from ..schemas.ai_explanation import AIExplanationRequest, AIExplanationResponse
from ..schemas.sentiment import AggregatedSentimentResponse
from ..schemas.volatility import VolatilityResponse
from ..schemas.event import EventResponse as EconomicEventSchemaResponse

logger = logging.getLogger(__name__)
settings = get_settings()

class AIExplanationService:
    """Generates AI-driven explanations based on queries and context."""

    def __init__(self, 
                 ml_model_factory: MLModelFactory = Depends(get_ml_model_factory),
                 market_service: MarketDataService = Depends(),
                 sentiment_service: SentimentAnalysisService = Depends(),
                 volatility_service: VolatilityService = Depends(),
                 calendar_service: EconomicCalendarService = Depends()
                 ):
        """Initialize the service with dependencies."""
        self.ml_model_factory = ml_model_factory
        self.market_service = market_service
        self.sentiment_service = sentiment_service
        self.volatility_service = volatility_service
        self.calendar_service = calendar_service

        if not self.ml_model_factory.llm:
            logger.error("MLModelFactory does not have an initialized LLM. AIExplanationService may not function fully.")

    async def _gather_asset_context(self, symbol: str, lookback_days: int = 7) -> Dict[str, Any]:
        """Gathers context for a specific asset symbol."""
        context = {"symbol": symbol}
        try:
            market_data_task = self.market_service.get_stock_data(symbol)
            sentiment_task = self.sentiment_service.get_aggregated_sentiment(symbol, date.today())
            volatility_task = self.volatility_service.calculate_and_predict_volatility(symbol)

            results = await asyncio.gather(
                market_data_task, sentiment_task, volatility_task,
                return_exceptions=True
            )
            
            market_data, sentiment, volatility = results[0], results[1], results[2]

            if not isinstance(market_data, Exception) and market_data:
                context["market_data"] = self._summarize_market_data(market_data)
            if not isinstance(sentiment, Exception) and sentiment:
                context["sentiment_data"] = self._summarize_sentiment(sentiment)
            if not isinstance(volatility, Exception) and volatility:
                context["volatility_data"] = self._summarize_volatility(volatility)

        except Exception as e:
            logger.error(f"Error gathering asset context for {symbol}: {e}", exc_info=True)
        return context

    async def _gather_event_context(self, event_id: int) -> Dict[str, Any]:
        """Gathers context for a specific economic event."""
        context = {"event_id": event_id}
        try:
            events = await self.calendar_service.get_economic_events(start_date=date.today() - timedelta(days=30), end_date=date.today() + timedelta(days=30))
            target_event_model: Optional[EconomicEventSchemaResponse] = None
            for ev_model in events:
                if ev_model.id == event_id:
                    target_event_model = ev_model
                    break
            
            if target_event_model:
                context["event_details"] = target_event_model.model_dump_json(indent=2)
            else:
                context["event_details"] = f"Event with ID {event_id} not found or details unavailable."

        except Exception as e:
            logger.error(f"Error gathering event context for event ID {event_id}: {e}", exc_info=True)
        return context

    async def generate_explanation(self, request: AIExplanationRequest) -> AIExplanationResponse:
        """Generates an AI explanation based on the user's query and context parameters."""
        logger.info(f"Generating AI explanation for query: '{request.query}' with params: {request.context_params}")

        if not self.ml_model_factory.llm:
            raise HTTPException(status_code=503, detail="AI Explanation Service is currently unavailable (LLM not configured).")

        context_data: Dict[str, Any] = {"query_timestamp": datetime.now().isoformat()}
        gathered_context_summary: Dict[str, Any] = {}

        if request.context_params:
            symbol = request.context_params.get("symbol")
            event_id = request.context_params.get("event_id")

            if symbol and isinstance(symbol, str):
                logger.info(f"Gathering asset-specific context for symbol: {symbol}")
                asset_context = await self._gather_asset_context(symbol)
                context_data.update(asset_context)
                gathered_context_summary["asset_context_for"] = symbol
                gathered_context_summary.update(asset_context)

            if event_id and isinstance(event_id, int):
                logger.info(f"Gathering event-specific context for event_id: {event_id}")
                event_context = await self._gather_event_context(event_id)
                context_data.update(event_context)
                gathered_context_summary["event_context_for"] = event_id
                gathered_context_summary.update(event_context)
        
        if not gathered_context_summary:
            gathered_context_summary["info"] = "No specific asset or event context provided; general query."

        explanation_str = await self.ml_model_factory.generate_ai_explanation(
            query=request.query,
            context_data=context_data
        )

        if explanation_str.startswith("Error:"):
            raise HTTPException(status_code=503, detail=explanation_str)

        return AIExplanationResponse(
            query=request.query,
            explanation=explanation_str,
            supporting_data_summary=gathered_context_summary,
            generated_at=datetime.now()
        )

    def _summarize_market_data(self, data: Optional[Dict | Exception]) -> str:
        if isinstance(data, Exception) or not data:
            return "Market data unavailable."
        try:
            price = data.get('current_price', 'N/A')
            change_pct = data.get('change_percent', 'N/A')
            volume = data.get('volume', 'N/A')
            summary = f"Current Price: {price}, Change: {change_pct}%, Volume: {volume}."
            return summary
        except Exception as e:
            logger.warning(f"Error summarizing market data: {e}", exc_info=True)
            return "Error summarizing market data."

    def _summarize_sentiment(self, data: Optional[AggregatedSentimentResponse | Exception]) -> str:
        if isinstance(data, Exception) or not data:
            return "Sentiment data unavailable."
        try:
            return (
                f"Overall Sentiment: {data.overall_sentiment.value}, Score: {data.normalized_score:.3f}. "
                f"News Score: {data.news_sentiment_score if data.news_sentiment_score is not None else 'N/A'}, "
                f"Reddit Score: {data.reddit_sentiment_score if data.reddit_sentiment_score is not None else 'N/A'}. "
                f"Benchmark: {data.benchmark}. Market Condition Context: {data.market_condition}."
            )
        except Exception as e:
            logger.warning(f"Error summarizing sentiment data: {e}", exc_info=True)
            return "Error summarizing sentiment data."

    def _summarize_volatility(self, data: Optional[Dict | Exception]) -> str:
        if isinstance(data, Exception) or not data:
            return "Volatility data unavailable."
        try:
            current_vol = data.get('current_volatility', 'N/A')
            predicted_vol = data.get('predicted_volatility', 'N/A')
            trend = data.get('trend', 'N/A')
            conditions = data.get('market_conditions', 'N/A')
            return (
                f"Current Volatility: {current_vol:.4f}, Predicted: {predicted_vol:.4f}. "
                f"Trend: {trend}. Market Conditions: {conditions}."
            )
        except Exception as e:
            logger.warning(f"Error summarizing volatility data: {e}", exc_info=True)
            return "Error summarizing volatility data."

    def _summarize_events(self, data: Optional[List[EconomicEventSchemaResponse] | Exception]) -> str:
        if isinstance(data, Exception) or not data:
            return "Economic events data unavailable."
        try:
            if not data:
                return "No notable upcoming economic events."
            summary_parts = []
            for event in data[:3]:
                summary_parts.append(f"{event.name} on {event.date.strftime('%Y-%m-%d')} (Impact: {event.impact}, AI Score: {event.impact_score or 'N/A'}).")
            return " Upcoming Events: " + " ".join(summary_parts)
        except Exception as e:
            logger.warning(f"Error summarizing economic events: {e}", exc_info=True)
            return "Error summarizing events data."

def get_ai_explanation_service(
    ml_model_factory: MLModelFactory = Depends(get_ml_model_factory),
    market_service: MarketDataService = Depends(get_market_data_service),
    sentiment_service: SentimentAnalysisService = Depends(get_sentiment_service),
    volatility_service: VolatilityService = Depends(get_volatility_service),
    calendar_service: EconomicCalendarService = Depends(get_economic_calendar_service) 
) -> AIExplanationService:
    return AIExplanationService(
        ml_model_factory=ml_model_factory,
        market_service=market_service,
        sentiment_service=sentiment_service,
        volatility_service=volatility_service,
        calendar_service=calendar_service
    )
