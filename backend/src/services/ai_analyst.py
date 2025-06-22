from fastapi import Depends, HTTPException
from datetime import datetime, date
import logging

from ..schemas.ai_analyst import AIAnalystResponse, Insight
from ..services.sentiment_analysis import SentimentAnalysisService
from ..services.volatility import VolatilityService
from ..core.dependencies import get_sentiment_service, get_volatility_service
from ..schemas.sentiment import AggregatedSentimentResponse
from ..schemas.volatility import VolatilityResponse as VolatilitySchemaResponse
from ..database.models import SentimentType

logger = logging.getLogger(__name__)

class AIAnalystService:
    def __init__(
        self,
        sentiment_service: SentimentAnalysisService,
        volatility_service: VolatilityService,
    ):
        self.sentiment_service = sentiment_service
        self.volatility_service = volatility_service

    async def get_insights(self, symbol: str) -> AIAnalystResponse:
        try:
            sentiment_data: AggregatedSentimentResponse
            try:
                sentiment_data_result = await self.sentiment_service.get_aggregated_sentiment(symbol)
                if not sentiment_data_result:
                    raise ValueError("Sentiment service returned empty data.")
                sentiment_data = sentiment_data_result
            except Exception as e:
                logger.error(f"Could not get sentiment data for {symbol}: {e}. Using fallback data.")
                sentiment_data = AggregatedSentimentResponse(
                    id=0,
                    symbol=symbol,
                    date=date.today(),
                    overall_sentiment=SentimentType.NEUTRAL,
                    normalized_score=0.0,
                    avg_daily_score=0.0,
                    moving_avg_7d=0.0,
                    benchmark="SPY",
                    news_sentiment_details={},
                    reddit_sentiment_details={},
                    news_sentiment_score=0.0,
                    reddit_sentiment_score=0.0,
                    market_condition="normal",
                    volatility_context={
                        "level": 0.2,
                        "is_high": False,
                        "trend": "stable"
                    },
                    source_weights={
                        "news_sentiment": 0.6,
                        "reddit_sentiment": 0.4
                    },
                    timestamp=datetime.now(),
                    data_quality_score=0.5,
                    sentiment_strength=0.0,
                    sentiment_confidence=0.5,
                    data_sources_available={"news": False, "reddit": False}
                )

            volatility_data: VolatilitySchemaResponse
            try:
                volatility_dict = await self.volatility_service.calculate_and_predict_volatility(symbol)
                if not volatility_dict:
                    raise ValueError("Volatility service returned empty data.")
                volatility_data = VolatilitySchemaResponse(**volatility_dict)
            except Exception as e:
                logger.error(f"Could not get volatility data for {symbol}: {e}. Using fallback data.")
                volatility_data = VolatilitySchemaResponse(
                    symbol=symbol,
                    timestamp=datetime.now(),
                    current_volatility=0.2,
                    historical_volatility_annualized=0.22,
                    volatility_10d_percentile=50.0,
                    predicted_volatility=0.21,
                    prediction_range={"low": 0.15, "high": 0.27},
                    market_conditions="normal",
                    volatility_regime="normal",
                    is_high_volatility=False,
                    trend="stable",
                    confidence_score=0.7,
                    metadata={
                        "model_version": "fallback_0.1",
                        "features_used": [],
                        "last_updated": datetime.now()
                    }
                )

            insights = self._generate_insights(sentiment_data, volatility_data)
            summary, outlook = self._summarize_insights(insights)

            return AIAnalystResponse(
                symbol=symbol,
                overall_outlook=outlook,
                summary=summary,
                insights=insights,
                generated_at=datetime.now()
            )
        except Exception as e:
            logger.exception(f"Error generating insights for {symbol}: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate AI analysis.")

    def _generate_insights(self, sentiment: AggregatedSentimentResponse, volatility: VolatilitySchemaResponse) -> list[Insight]:
        insights = []

        # Volatility Insights
        if volatility.is_high_volatility:
            insights.append(Insight(
                title="High Volatility Warning",
                description=f"The stock is experiencing high volatility ({volatility.current_volatility:.2f}). Expect larger than usual price swings.",
                confidence=0.9,
                category="Volatility"
            ))
        if "increasing" in volatility.trend:
            insights.append(Insight(
                title="Rising Volatility",
                description="Volatility is trending upwards, suggesting increasing market uncertainty.",
                confidence=0.8,
                category="Volatility"
            ))

        # Sentiment Insights
        if sentiment.overall_sentiment.value == "bullish" and sentiment.normalized_score > 0.5:
            insights.append(Insight(
                title="Strong Bullish Sentiment",
                description=f"Market sentiment is strongly bullish with a score of {sentiment.normalized_score:.2f}, driven by positive news and social media conversation.",
                confidence=sentiment.sentiment_confidence or 0.85,
                category="Sentiment"
            ))
        elif sentiment.overall_sentiment.value == "bearish" and sentiment.normalized_score < -0.5:
             insights.append(Insight(
                title="Strong Bearish Sentiment",
                description=f"Market sentiment is strongly bearish with a score of {sentiment.normalized_score:.2f}, driven by negative news and social media conversation.",
                confidence=sentiment.sentiment_confidence or 0.85,
                category="Sentiment"
            ))

        # Combined Insights
        if volatility.is_high_volatility and sentiment.overall_sentiment.value == "bullish":
            insights.append(Insight(
                title="Opportunity: Bullish High-Volatility",
                description="High volatility combined with strong bullish sentiment may present trading opportunities for those with high risk tolerance.",
                confidence=0.75,
                category="Opportunity"
            ))
        elif volatility.is_high_volatility and sentiment.overall_sentiment.value == "bearish":
            insights.append(Insight(
                title="Warning: Bearish High-Volatility",
                description="High volatility combined with strong bearish sentiment is a risky environment. Caution is advised.",
                confidence=0.75,
                category="Opportunity"
            ))
        
        if not insights:
            insights.append(Insight(
                title="Neutral Conditions",
                description="Market conditions appear stable and sentiment is neutral. No strong signals detected.",
                confidence=0.9,
                category="General"
            ))

        return insights
    
    def _summarize_insights(self, insights: list[Insight]) -> (str, str):
        if not insights:
            return "No significant insights found.", "Neutral"

        summary = "Key insights: " + ", ".join([i.title for i in insights[:2]]) + "."
        
        outlook = "Neutral"
        if any("Bullish" in i.title for i in insights):
            outlook = "Cautiously Optimistic"
        if any("Bearish" in i.title for i in insights):
            outlook = "Cautiously Pessimistic"
        if any("Bullish" in i.title for i in insights) and any("High Volatility" in i.title for i in insights):
            outlook = "Volatile Opportunity"

        return summary, outlook

def get_ai_analyst_service(
    sentiment_service: SentimentAnalysisService = Depends(get_sentiment_service),
    volatility_service: VolatilityService = Depends(get_volatility_service),
) -> AIAnalystService:
    return AIAnalystService(sentiment_service, volatility_service) 