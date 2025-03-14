from typing import List, Dict, Any, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from ..database.models import User, Alert
import logging
import asyncio
from .market_data import MarketDataService
from .sentiment_analysis import SentimentAnalysisService
from .volatility import VolatilityService

logger = logging.getLogger(__name__)

class AlertService:
    def __init__(self, db: Session):
        self.db = db
        self.market_service = MarketDataService()
        self.sentiment_service = SentimentAnalysisService()
        self.volatility_service = VolatilityService()
        self.alert_conditions = {
            "price": self._check_price_condition,
            "volatility": self._check_volatility_condition,
            "sentiment": self._check_sentiment_condition
        }

    async def create_alert(
        self,
        user_id: int,
        symbol: str,
        alert_type: str,
        condition: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create a new alert for a user."""
        new_alert = Alert(
            user_id=user_id,
            symbol=symbol,
            alert_type=alert_type,
            condition=condition,
            created_at=datetime.now(),
            is_active=True,
            last_checked=datetime.now()
        )
        
        self.db.add(new_alert)
        self.db.commit()
        self.db.refresh(new_alert)
        
        return {
            "id": new_alert.id,
            "user_id": new_alert.user_id,
            "symbol": new_alert.symbol,
            "alert_type": new_alert.alert_type,
            "condition": new_alert.condition,
            "created_at": new_alert.created_at,
            "is_active": new_alert.is_active
        }

    async def check_alert_conditions(self, alert: Dict[str, Any]) -> bool:
        """Check if alert conditions are met."""
        check_func = self.alert_conditions.get(alert["alert_type"])
        if not check_func:
            return False
        return await check_func(alert["symbol"], alert["condition"])

    async def _check_price_condition(
        self,
        symbol: str,
        condition: Dict[str, Any]
    ) -> bool:
        data = await self.market_service.get_stock_data(symbol)
        current_price = float(data["price"])
        
        if condition["type"] == "above" and current_price > condition["value"]:
            return True
        if condition["type"] == "below" and current_price < condition["value"]:
            return True
        return False

    async def _check_volatility_condition(
        self,
        symbol: str,
        condition: Dict[str, Any]
    ) -> bool:
        volatility = await self.volatility_service.calculate_volatility(symbol)
        current_vol = volatility["historical_volatility"]
        
        if condition["type"] == "above" and current_vol > condition["value"]:
            return True
        if condition["type"] == "below" and current_vol < condition["value"]:
            return True
        return False

    async def _check_sentiment_condition(
        self,
        symbol: str,
        condition: Dict[str, Any]
    ) -> bool:
        sentiment = await self.sentiment_service.get_market_sentiment(symbol)
        return sentiment["sentiment"] == condition["sentiment"]

    async def get_user_alerts(self, user_id: int) -> List[Dict[str, Any]]:
        """Get all alerts for a user."""
        alerts = self.db.query(Alert).filter(Alert.user_id == user_id).all()
        return [
            {
                "id": alert.id,
                "user_id": alert.user_id,
                "symbol": alert.symbol,
                "alert_type": alert.alert_type,
                "condition": alert.condition,
                "created_at": alert.created_at,
                "is_active": alert.is_active,
                "last_checked": alert.last_checked,
                "last_triggered": alert.last_triggered
            }
            for alert in alerts
        ]

    async def delete_alert(self, alert_id: int, user_id: int) -> bool:
        """Delete an alert."""
        alert = self.db.query(Alert).filter(
            Alert.id == alert_id,
            Alert.user_id == user_id
        ).first()
        
        if not alert:
            return False
            
        self.db.delete(alert)
        self.db.commit()
        return True
