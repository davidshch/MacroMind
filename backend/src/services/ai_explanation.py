from typing import Dict, Any, List
import logging
from datetime import datetime
from .market_data import MarketDataService
from .sentiment_analysis import SentimentAnalysisService
from .volatility import VolatilityService
from .economic_calendar import EconomicCalendarService

logger = logging.getLogger(__name__)

class AIExplanationService:
    def __init__(self):
        self.market_service = MarketDataService()
        self.sentiment_service = SentimentAnalysisService()
        self.volatility_service = VolatilityService()
        self.calendar_service = EconomicCalendarService()

    async def explain_market_conditions(self, symbol: str) -> Dict[str, Any]:
        """Generate comprehensive market explanation."""
        try:
            # Gather all relevant data
            market_data = await self.market_service.get_stock_data(symbol)
            sentiment = await self.sentiment_service.get_market_sentiment(symbol)
            volatility = await self.volatility_service.calculate_volatility(symbol)
            events = await self.calendar_service.get_economic_events()

            # Generate natural language explanation
            explanation = self._generate_market_explanation(
                symbol, market_data, sentiment, volatility, events
            )

            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "explanation": explanation,
                "key_factors": self._identify_key_factors(
                    sentiment, volatility, events
                ),
                "risk_assessment": self._assess_risk(
                    sentiment, volatility
                ),
                "action_suggestions": self._generate_suggestions(
                    sentiment, volatility, events
                )
            }
        except Exception as e:
            logger.error(f"Error generating explanation: {str(e)}")
            raise

    def _generate_market_explanation(
        self, 
        symbol: str,
        market_data: Dict[str, Any],
        sentiment: Dict[str, Any],
        volatility: Dict[str, Any],
        events: List[Dict[str, Any]]
    ) -> str:
        """Generate natural language explanation of market conditions."""
        price = market_data.get("price", "N/A")
        sentiment_label = sentiment.get("sentiment", "neutral")
        vol_condition = volatility.get("market_conditions", "normal")
        
        explanation = f"Analysis for {symbol}: The stock is trading at ${price} with "
        explanation += f"a {sentiment_label} market sentiment. "
        explanation += f"Market conditions show {vol_condition} volatility levels. "
        
        if events:
            explanation += "\n\nUpcoming market events that may impact the stock: "
            for event in events[:2]:
                explanation += f"\n- {event['event']} on {event['date']}"
        
        return explanation

    def _identify_key_factors(
        self,
        sentiment: Dict[str, Any],
        volatility: Dict[str, Any],
        events: List[Dict[str, Any]]
    ) -> List[str]:
        """Identify key factors affecting the stock."""
        factors = []
        
        if sentiment["confidence"] > 0.8:
            factors.append(f"Strong {sentiment['sentiment']} sentiment")
        
        if volatility["market_conditions"] != "normal":
            factors.append(f"{volatility['market_conditions'].replace('_', ' ').title()}")
        
        for event in events[:2]:
            if event["importance"] == "high":
                factors.append(f"Upcoming {event['event']}")
                
        return factors

    def _assess_risk(
        self,
        sentiment: Dict[str, Any],
        volatility: Dict[str, Any]
    ) -> str:
        """Assess overall market risk."""
        if volatility["market_conditions"] == "highly_volatile" and sentiment["sentiment"] == "bearish":
            return "High Risk"
        elif volatility["market_conditions"] == "low_volatility" and sentiment["sentiment"] == "bullish":
            return "Low Risk"
        return "Moderate Risk"

    def _generate_suggestions(
        self,
        sentiment: Dict[str, Any],
        volatility: Dict[str, Any],
        events: List[Dict[str, Any]]
    ) -> List[str]:
        """Generate action suggestions based on analysis."""
        suggestions = []
        
        if volatility["market_conditions"] == "highly_volatile":
            suggestions.append("Consider reducing position size due to high volatility")
        
        if sentiment["sentiment"] == "bearish" and sentiment["confidence"] > 0.8:
            suggestions.append("Watch for potential downside movement")
        elif sentiment["sentiment"] == "bullish" and sentiment["confidence"] > 0.8:
            suggestions.append("Look for potential entry points")
            
        if events:
            suggestions.append(f"Monitor upcoming {events[0]['event']} for market impact")
            
        return suggestions
