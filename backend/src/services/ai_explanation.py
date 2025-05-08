"""AI-powered market analysis and explanation service using LangChain and OpenAI."""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime, date
import asyncio
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema import SystemMessage, HumanMessage
from fastapi import HTTPException, Depends

from ..config import get_settings
from .market_data import MarketDataService
from .sentiment_analysis import SentimentAnalysisService
from .volatility import VolatilityService
from .economic_calendar import EconomicCalendarService
from ..schemas.sentiment import AggregatedSentimentResponse
from ..schemas.volatility import VolatilityResponse
from ..schemas.event import EventResponse
from ..database.database import get_db
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)
settings = get_settings()

LLM_MODEL = "gpt-4o"

EXPLANATION_PROMPT_TEMPLATE = """You are MacroMind AI, a sophisticated financial analyst AI. Your task is to provide a clear, concise, and insightful market explanation for the symbol {symbol} based on the provided data. Structure your response as a JSON object with the following keys: "explanation", "key_factors", "risk_assessment", "action_suggestions".

**Input Data:**

*   **Symbol:** {symbol}
*   **Date:** {current_date}
*   **Market Data:** {market_data_summary}
*   **Aggregated Sentiment:** {sentiment_summary}
*   **Volatility Analysis:** {volatility_summary}
*   **Upcoming Economic Events:** {events_summary}

**Instructions:**

1.  **explanation:** Write a brief (3-5 sentences) narrative summarizing the current market situation for the symbol. Integrate price action, sentiment, and volatility.
2.  **key_factors:** List the top 3-4 most significant factors currently influencing the symbol (e.g., "Strong bullish sentiment", "High volatility regime", "Upcoming CPI report").
3.  **risk_assessment:** Provide a risk level (Low, Moderate-Low, Moderate, Moderate-High, High) based on the interplay of sentiment and volatility. Briefly justify the assessment.
4.  **action_suggestions:** Offer 2-3 actionable insights or areas to monitor (e.g., "Consider risk management due to high volatility", "Monitor upcoming Fed meeting", "Look for confirmation of bullish trend").

**Output Format:** Return ONLY the JSON object, with no other text before or after it.

```json
{{
    "explanation": "<Your generated explanation>",
    "key_factors": [
        "<Factor 1>",
        "<Factor 2>",
        "<Factor 3>"
    ],
    "risk_assessment": "<Risk Level (e.g., Moderate)>: <Brief Justification>",
    "action_suggestions": [
        "<Suggestion 1>",
        "<Suggestion 2>"
    ]
}}
```
"""

class AIExplanationService:
    """Generates market explanations using LangChain and an LLM."""

    def __init__(self, 
                 market_service: MarketDataService = Depends(),
                 sentiment_service: SentimentAnalysisService = Depends(),
                 volatility_service: VolatilityService = Depends(),
                 calendar_service: EconomicCalendarService = Depends()
                 ):
        """Initialize the service with dependencies and LangChain components."""
        if not settings.openai_api_key:
            logger.error("OpenAI API key not configured. AIExplanationService cannot function.")
            raise ValueError("OpenAI API key is required for AIExplanationService.")
        
        self.market_service = market_service
        self.sentiment_service = sentiment_service
        self.volatility_service = volatility_service
        self.calendar_service = calendar_service

        try:
            self.llm = ChatOpenAI(temperature=0.3, model_name=LLM_MODEL, openai_api_key=settings.openai_api_key)
        except Exception as e:
            logger.error(f"Failed to initialize LangChain components: {e}")
            raise RuntimeError("Could not initialize AI Explanation Service LLM.") from e

    async def explain_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """
        Generate a comprehensive market explanation using an LLM.

        Args:
            symbol: The stock or asset symbol.

        Returns:
            A dictionary containing the LLM-generated explanation, factors, risk,
            and suggestions, plus data status.

        Raises:
            HTTPException: If data fetching or LLM generation fails.
        """
        logger.info(f"Generating LLM market explanation for symbol: {symbol}")
        try:
            market_data_task = self.market_service.get_stock_data(symbol)
            sentiment_task = self.sentiment_service.get_aggregated_sentiment(symbol, date.today())
            volatility_task = self.volatility_service.calculate_and_predict_volatility(symbol)
            events_task = self.calendar_service.get_upcoming_events(days_ahead=7)

            market_data, sentiment, volatility, all_events = await asyncio.gather(
                market_data_task, sentiment_task, volatility_task, events_task,
                return_exceptions=True
            )

            market_data_summary = self._summarize_market_data(market_data)
            sentiment_summary = self._summarize_sentiment(sentiment)
            volatility_summary = self._summarize_volatility(volatility)
            events_summary = self._summarize_events(all_events)

            if isinstance(market_data, Exception) and isinstance(sentiment, Exception) and isinstance(volatility, Exception):
                 logger.error(f"All data sources failed for {symbol}. Cannot generate explanation.")
                 raise HTTPException(status_code=503, detail="Failed to fetch required data for explanation.")

            prompt_content = EXPLANATION_PROMPT_TEMPLATE.format(
                symbol=symbol,
                current_date=date.today().isoformat(),
                market_data_summary=market_data_summary,
                sentiment_summary=sentiment_summary,
                volatility_summary=volatility_summary,
                events_summary=events_summary
            )
            
            messages = [
                SystemMessage(content="You are MacroMind AI, a sophisticated financial analyst AI. Respond ONLY with the requested JSON object."),
                HumanMessage(content=prompt_content)
            ]

            logger.debug(f"Invoking LLM for {symbol} explanation...")
            response = await self.llm.ainvoke(messages)
            response_str = response.content
            logger.debug(f"LLM raw response for {symbol}: {response_str}")

            try:
                if response_str.startswith("```json"):
                    response_str = response_str.strip("` \njson")
                explanation_json = json.loads(response_str)
            except json.JSONDecodeError as json_err:
                logger.error(f"Failed to parse LLM JSON response for {symbol}: {json_err}. Response: {response_str}")
                raise HTTPException(status_code=500, detail="AI explanation generation failed (parsing error).")

            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                **explanation_json,
                "data_sources_status": {
                    "market_data": "OK" if not isinstance(market_data, Exception) else "Failed",
                    "sentiment": "OK" if not isinstance(sentiment, Exception) else "Failed",
                    "volatility": "OK" if not isinstance(volatility, Exception) else "Failed",
                    "events": "OK" if not isinstance(all_events, Exception) else "Failed"
                }
            }
        except HTTPException as he:
            raise he
        except Exception as e:
            logger.exception(f"Error generating LLM explanation for {symbol}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Could not generate AI explanation for {symbol}.")

    def _summarize_market_data(self, data: Optional[Dict | Exception]) -> str:
        if isinstance(data, Exception) or not data:
            return "Market data unavailable."
        try:
            pd = data.get('price_data', {})
            price = pd.get('current_price', 'N/A')
            change = pd.get('change_percent', 'N/A')
            vol = pd.get('volume', 'N/A')
            summary = f"Price: {price}, Change: {change}%, Volume: {vol}."
            metrics = data.get('metrics', {})
            pe = metrics.get('pe_ratio', 'N/A')
            beta = metrics.get('beta', 'N/A')
            if pe != 'N/A' or beta != 'N/A':
                summary += f" P/E: {pe}, Beta: {beta}."
            return summary
        except Exception as e:
            logger.warning(f"Error summarizing market data: {e}")
            return "Error summarizing market data."

    def _summarize_sentiment(self, data: Optional[AggregatedSentimentResponse | Exception]) -> str:
        if isinstance(data, Exception) or not data:
            return "Sentiment data unavailable."
        try:
            return (f"Overall: {data.overall_sentiment.value}, Score: {data.normalized_score:.3f}. "
                    f"Market Condition Context: {data.market_condition}. "
                    f"Sources: {list(data.source_weights.keys())}.")
        except Exception as e:
            logger.warning(f"Error summarizing sentiment data: {e}")
            return "Error summarizing sentiment data."

    def _summarize_volatility(self, data: Optional[VolatilityResponse | Exception]) -> str:
        if isinstance(data, Exception) or not data:
            return "Volatility data unavailable."
        try:
            return (f"Current: {data.current_volatility:.4f}, Predicted: {data.predicted_volatility:.4f} "
                    f"(Range: {data.prediction_range.low:.4f}-{data.prediction_range.high:.4f}). "
                    f"Regime: {data.volatility_regime.value}, Trend: {data.trend}. "
                    f"High Volatility: {data.is_high_volatility}.")
        except Exception as e:
            logger.warning(f"Error summarizing volatility data: {e}")
            return "Error summarizing volatility data."

    def _summarize_events(self, data: Optional[List[EventResponse] | Exception]) -> str:
        if isinstance(data, Exception):
            return "Economic event data unavailable."
        if not data:
            return "No major relevant economic events found in the near term."
        try:
            summary = "Upcoming Events: "
            event_strs = [f"{e.name} ({e.date.strftime('%Y-%m-%d') if e.date else 'N/A'}, Impact: {e.impact or 'N/A'})" for e in data[:3]]
            return summary + "; ".join(event_strs) + "."
        except Exception as e:
            logger.warning(f"Error summarizing event data: {e}")
            return "Error summarizing event data."

def get_market_data_service() -> MarketDataService:
    return MarketDataService()

def get_volatility_service() -> VolatilityService:
    return VolatilityService()

def get_economic_calendar_service(db: Session = Depends(get_db)) -> EconomicCalendarService:
    return EconomicCalendarService(db=db)

def get_sentiment_analysis_service(db: Session = Depends(get_db)) -> SentimentAnalysisService:
    return SentimentAnalysisService(db=db)

def get_ai_explanation_service(
    market_service: MarketDataService = Depends(get_market_data_service),
    sentiment_service: SentimentAnalysisService = Depends(get_sentiment_analysis_service),
    volatility_service: VolatilityService = Depends(get_volatility_service),
    calendar_service: EconomicCalendarService = Depends(get_economic_calendar_service)
) -> AIExplanationService:
    return AIExplanationService(
        market_service=market_service,
        sentiment_service=sentiment_service,
        volatility_service=volatility_service,
        calendar_service=calendar_service
    )
