from fastapi import Depends, HTTPException
from datetime import datetime, date
import logging
import hashlib
import random

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
        logger.info(f"DEMO MODE: Returning high-quality static AI analysis for symbol '{symbol}'.")
        try:
            symbol_upper = symbol.upper()
            base_confidence = 0.85
            outlook = "Neutral to Cautiously Bullish, with High Volatility"
            summary = (
                f"The AI analysis for {symbol_upper} indicates a period of high uncertainty. While there is potential for a bullish reversal following the recent dip, "
                f"volatility is expected to be high and sentiment signals are mixed. Key external factors, such as sector-wide news, could be significant drivers in the short term."
            )

            if symbol_upper == "TSLA":
                base_confidence = 0.80
                outlook = "Volatile with Bearish Undercurrents"
                summary = (
                    f"AI analysis for {symbol_upper} reveals a highly volatile environment. While the stock is known for sharp movements, "
                    f"current social media sentiment is leaning bearish, and the elevated volatility prediction suggests significant downside risk. "
                    f"Any company-specific news could act as a major catalyst."
                )
            elif symbol_upper == "NVDA":
                base_confidence = 0.90
                outlook = "Strongly Bullish, but Watch for Profit-Taking"
                summary = (
                    f"The outlook for {symbol_upper} is strongly bullish, backed by overwhelming positive sentiment from both news and social media. "
                    f"However, predicted volatility is rising, which could indicate profit-taking or a short-term consolidation period. "
                    f"The primary trend remains upward."
                )
            else:
                # Generic but varied generation based on symbol hash to avoid identical output.
                hash_int = int(hashlib.sha256(symbol_upper.encode()).hexdigest(), 16)
                random.seed(hash_int)

                outlook_options = [
                    "Stable with Mild Upside Potential",
                    "Range-Bound and Awaiting Catalyst",
                    "Cautiously Bearish with Low Volatility",
                ]
                outlook = random.choice(outlook_options)

                summary_templates = [
                    f"{symbol_upper} is currently trading in a tight range with limited directional bias. Key technical levels are acting as magnets while broader market sentiment remains neutral.",
                    f"The market seems undecided on {symbol_upper}. Despite mixed news flow, option markets are pricing in modest moves, suggesting participants are waiting for a breakout trigger.",
                    f"{symbol_upper} shows no strong momentum either way. Fundamental drivers appear priced in, so traders may focus on short-term mean-reversion strategies until new information emerges.",
                ]
                summary = random.choice(summary_templates)

                base_confidence = round(0.75 + (hash_int % 10) * 0.01, 2)  # 0.75-0.84

            # Build varied insight titles/descriptions deterministically per symbol
            def pick(options: list[str]) -> str:
                return random.choice(options)

            vol_titles = [
                "Elevated Volatility Warning",
                "Volatility Cooling Off",
                "Sudden Volatility Spike",
                "Volatility Compression Phase"
            ]
            sent_titles = [
                "Mixed Sentiment Signals",
                "Bullish Social Buzz",
                "Bearish News Flow",
                "Diverging Retail vs Institutional Sentiment"
            ]
            opp_titles = [
                "Post-Dip Consolidation Opportunity",
                "Breakout Setup Forming",
                "Mean-Reversion Play",
                "Momentum Continuation Setup"
            ]
            driver_titles = [
                "Key Driver: Sector-Wide Scrutiny",
                "Key Driver: Macro Headwinds",
                "Key Driver: Earnings Anticipation",
                "Key Driver: Regulatory Catalysts"
            ]

            demo_insights = [
                Insight(
                    title=pick(vol_titles),
                    description=f"Predicted volatility for {symbol_upper} is currently {'above' if base_confidence>0.8 else 'near'} its recent average. Expect {{'larger' if base_confidence>0.8 else 'modest'}} price swings in the coming sessions.",
                    confidence=round(base_confidence + 0.05, 2),
                    category="Volatility"
                ),
                Insight(
                    title=pick(sent_titles),
                    description=f"Sentiment readings for {symbol_upper} show a {{'cautiously optimistic' if base_confidence>0.8 else 'neutral to slightly bearish'}} tone across major channels.",
                    confidence=round(base_confidence, 2),
                    category="Sentiment"
                ),
                Insight(
                    title=pick(opp_titles),
                    description=f"Price action in {symbol_upper} suggests a potential trading setup. Watch the {{'resistance' if base_confidence>0.8 else 'support'}} levels for confirmation.",
                    confidence=round(base_confidence - 0.1, 2),
                    category="Opportunity"
                ),
                Insight(
                    title=pick(driver_titles),
                    description=f"An upcoming catalyst specific to {symbol_upper} or its sector could override current technical signals. Stay alert to breaking headlines and macro releases.",
                    confidence=round(base_confidence - 0.05, 2),
                    category="Market"
                )
            ]

            return AIAnalystResponse(
                symbol=symbol_upper,
                generated_at=datetime.now(),
                overall_outlook=outlook,
                summary=summary,
                insights=demo_insights
            )
        except Exception as e:
            logger.exception(f"Error generating static demo insights for {symbol}: {e}")
            raise HTTPException(status_code=500, detail="Failed to generate AI analysis even in demo mode.")

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
        if str(sentiment.overall_sentiment) == "bullish" and sentiment.normalized_score > 0.5:
            insights.append(Insight(
                title="Strong Bullish Sentiment",
                description=f"Market sentiment is strongly bullish with a score of {sentiment.normalized_score:.2f}, driven by positive news and social media conversation.",
                confidence=sentiment.sentiment_confidence or 0.85,
                category="Sentiment"
            ))
        elif str(sentiment.overall_sentiment) == "bearish" and sentiment.normalized_score < -0.5:
             insights.append(Insight(
                title="Strong Bearish Sentiment",
                description=f"Market sentiment is strongly bearish with a score of {sentiment.normalized_score:.2f}, driven by negative news and social media conversation.",
                confidence=sentiment.sentiment_confidence or 0.85,
                category="Sentiment"
            ))

        # Combined Insights
        if volatility.is_high_volatility and str(sentiment.overall_sentiment) == "bullish":
            insights.append(Insight(
                title="Opportunity: Bullish High-Volatility",
                description="High volatility combined with strong bullish sentiment may present trading opportunities for those with high risk tolerance.",
                confidence=0.75,
                category="Opportunity"
            ))
        elif volatility.is_high_volatility and str(sentiment.overall_sentiment) == "bearish":
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