"""
Service layer for managing and evaluating alerts.
"""
import logging
from typing import List, Optional, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy import delete, update
from fastapi import HTTPException, Depends
from datetime import datetime, date
import asyncio

from src.database.models import Alert as AlertModel, User as UserModel
from src.schemas.alert import (
    AlertCreate,
    AlertResponse,
    AlertUpdate,
    AlertConditions,
    AlertConditionField,
    AlertConditionOperator,
)
from src.services.sentiment_analysis import SentimentAnalysisService
from src.services.volatility import VolatilityService
from .websocket import websocket_service_instance

from ..database.database import get_db
from ..core.dependencies import get_sentiment_service, get_volatility_service

logger = logging.getLogger(__name__)

class AlertService:
    def __init__(
        self,
        db: AsyncSession,
        sentiment_service: SentimentAnalysisService,
        volatility_service: VolatilityService,
    ):
        self.db = db
        self.sentiment_service = sentiment_service
        self.volatility_service = volatility_service

    async def create_alert(self, user_id: int, alert_data: AlertCreate) -> AlertModel:
        """
        Creates a new alert for a user.

        Args:
            user_id: The ID of the user creating the alert.
            alert_data: The data for the new alert.

        Returns:
            The created AlertModel object.

        Raises:
            HTTPException: If the user is not found.
        """
        user = await self.db.get(UserModel, user_id)
        if not user:
            logger.warning(f"User not found with id: {user_id} during alert creation.")
            raise HTTPException(status_code=404, detail="User not found")

        db_alert = AlertModel(
            user_id=user_id,
            name=alert_data.name,
            symbol=alert_data.symbol,
            conditions=alert_data.conditions.model_dump(),  # Store as dict/JSON
            notes=alert_data.notes,  # Add the notes field back
            is_active=alert_data.is_active,
        )
        self.db.add(db_alert)
        await self.db.commit()
        await self.db.refresh(db_alert)
        logger.info(f"Alert created: {db_alert.id} for user {user_id} on symbol {alert_data.symbol}")
        return db_alert

    async def get_alert_by_id(self, alert_id: int, user_id: int) -> Optional[AlertModel]:
        """
        Retrieves a specific alert by its ID for a given user.

        Args:
            alert_id: The ID of the alert to retrieve.
            user_id: The ID of the user who owns the alert.

        Returns:
            The AlertModel object if found, otherwise None.
        """
        stmt = select(AlertModel).where(AlertModel.id == alert_id, AlertModel.user_id == user_id)
        result = await self.db.execute(stmt)
        alert = result.scalar_one_or_none()
        if alert:
            logger.debug(f"Alert retrieved: {alert_id} for user {user_id}")
        else:
            logger.debug(f"Alert not found: {alert_id} for user {user_id}")
        return alert

    async def get_user_alerts(self, user_id: int, skip: int = 0, limit: int = 100) -> List[AlertModel]:
        """
        Retrieves all alerts for a specific user.

        Args:
            user_id: The ID of the user.
            skip: Number of records to skip for pagination.
            limit: Maximum number of records to return.

        Returns:
            A list of AlertModel objects.
        """
        stmt = select(AlertModel).where(AlertModel.user_id == user_id).offset(skip).limit(limit)
        result = await self.db.execute(stmt)
        alerts = result.scalars().all()
        logger.debug(f"Retrieved {len(alerts)} alerts for user {user_id}")
        return list(alerts)

    async def update_alert(
        self, alert_id: int, user_id: int, alert_update_data: AlertUpdate
    ) -> Optional[AlertModel]:
        """
        Updates an existing alert for a user.

        Args:
            alert_id: The ID of the alert to update.
            user_id: The ID of the user who owns the alert.
            alert_update_data: The data to update the alert with.

        Returns:
            The updated AlertModel object if found and updated, otherwise None.
        """
        alert = await self.get_alert_by_id(alert_id, user_id)
        if not alert:
            logger.warning(f"Alert not found for update: {alert_id}, user {user_id}")
            return None

        update_data = alert_update_data.model_dump(exclude_unset=True)

        # Ensure conditions are properly serialized if present in update_data
        if "conditions" in update_data and update_data["conditions"] is not None:
            if isinstance(update_data["conditions"], AlertConditions):
                update_data["conditions"] = update_data["conditions"].model_dump()
            elif not isinstance(update_data["conditions"], dict):
                logger.error(f"Invalid type for conditions in update_alert: {type(update_data['conditions'])}")
                raise HTTPException(status_code=400, detail="Invalid format for alert conditions.")

        if not update_data:
            logger.info(f"No update data provided for alert {alert_id}, user {user_id}")
            return alert

        stmt = (
            update(AlertModel)
            .where(AlertModel.id == alert_id, AlertModel.user_id == user_id)
            .values(**update_data)
            .returning(AlertModel)
        )

        # Execute and commit
        result = await self.db.execute(stmt)
        await self.db.commit()

        # Try to get the updated alert from the result
        updated_alert_instance = result.scalar_one_or_none()

        if updated_alert_instance:
            logger.info(f"Alert updated: {alert_id} for user {user_id}")
            return updated_alert_instance
        else:
            logger.warning(f"Alert update for {alert_id} did not return an instance directly, re-fetching.")
            refetched_alert = await self.get_alert_by_id(alert_id, user_id)
            if refetched_alert:
                logger.info(f"Successfully re-fetched and confirmed update for alert {alert_id}")
                return refetched_alert
            else:
                logger.error(f"Failed to update or re-fetch alert {alert_id} for user {user_id}. It might have been deleted.")
                return None

    async def delete_alert(self, alert_id: int, user_id: int) -> bool:
        """
        Deletes an alert for a user.

        Args:
            alert_id: The ID of the alert to delete.
            user_id: The ID of the user who owns the alert.

        Returns:
            True if the alert was deleted, False otherwise.
        """
        stmt = delete(AlertModel).where(AlertModel.id == alert_id, AlertModel.user_id == user_id)
        result = await self.db.execute(stmt)
        await self.db.commit()
        if result.rowcount > 0:
            logger.info(f"Alert deleted: {alert_id} for user {user_id}")
            return True
        logger.warning(f"Alert not found for deletion: {alert_id}, user {user_id}")
        return False

    async def _get_current_metric_value(self, symbol: str, metric_path: str) -> Optional[Any]:
        """Fetches the current value for a given metric path (e.g., 'sentiment.score')."""
        parts = metric_path.lower().split('.')
        data_type = parts[0]
        metric_name = parts[1] if len(parts) > 1 else None

        current_value = None
        try:
            if data_type == "sentiment":
                sentiment_data = await self.sentiment_service.get_aggregated_sentiment(symbol, target_date=date.today())
                if sentiment_data and metric_name:
                    if metric_name == "score" and hasattr(sentiment_data, 'normalized_score'):
                        current_value = sentiment_data.normalized_score
                    elif metric_name == "moving_avg_7d" and hasattr(sentiment_data, 'moving_avg_7d'):
                        current_value = sentiment_data.moving_avg_7d
                    else:
                        logger.warning(f"Unsupported sentiment metric: {metric_name} for alert on {symbol}")
            elif data_type == "volatility":
                volatility_data = await self.volatility_service.calculate_and_predict_volatility(symbol)
                if volatility_data and metric_name:
                    if metric_name == "predicted" and "predicted_volatility" in volatility_data:
                        current_value = volatility_data["predicted_volatility"]
                    elif metric_name == "current" and "current_volatility" in volatility_data:
                        current_value = volatility_data["current_volatility"]
                    elif metric_name == "trend" and "trend" in volatility_data:
                        current_value = volatility_data["trend"]
                    else:
                        logger.warning(f"Unsupported volatility metric: {metric_name} for alert on {symbol}")
            else:
                logger.warning(f"Unsupported data type for alert condition: {data_type} in metric {metric_path}")

            return current_value
        except Exception as e:
            logger.error(f"Error fetching current value for metric {metric_path} on symbol {symbol}: {e}", exc_info=True)
            return None

    async def _evaluate_single_condition(self, symbol: str, condition: AlertConditionField) -> bool:
        """Evaluates a single alert condition after fetching the current metric value."""
        current_value = await self._get_current_metric_value(symbol, condition.metric)

        if current_value is None:
            logger.debug(f"Could not retrieve current value for metric {condition.metric} on symbol {symbol}. Condition fails.")
            return False

        target_value = condition.value
        if isinstance(current_value, str):
            if condition.operator == AlertConditionOperator.EQUALS:
                return current_value.lower() == str(target_value).lower()
            else:
                logger.warning(f"Operator {condition.operator} not directly supported for string metric {condition.metric}. Condition fails.")
                return False

        if not isinstance(current_value, (int, float)):
            logger.warning(f"Current value for {condition.metric} is not numeric ({type(current_value)}). Condition fails.")
            return False

        try:
            target_value_float = float(target_value)
        except ValueError:
            logger.error(f"Condition value '{target_value}' for metric {condition.metric} is not a valid number. Condition fails.")
            return False

        op = condition.operator
        if op == AlertConditionOperator.GREATER_THAN:
            return current_value > target_value_float
        if op == AlertConditionOperator.LESS_THAN:
            return current_value < target_value_float
        if op == AlertConditionOperator.EQUALS:
            return abs(current_value - target_value_float) < 1e-9
        if op == AlertConditionOperator.GREATER_THAN_OR_EQUAL_TO:
            return current_value >= target_value_float
        if op == AlertConditionOperator.LESS_THAN_OR_EQUAL_TO:
            return current_value <= target_value_float

        logger.warning(f"Unsupported operator: {op} for alert condition on {symbol}")
        return False

    async def _evaluate_alert_conditions(self, alert: AlertModel) -> bool:
        """Evaluates all conditions for a given alert."""
        if not isinstance(alert.conditions, dict):
            logger.error(f"Alert {alert.id} has malformed conditions (not a dict): {alert.conditions}")
            return False

        try:
            conditions_schema = AlertConditions.model_validate(alert.conditions)
        except Exception as e:
            logger.error(f"Failed to validate alert conditions for alert {alert.id}: {alert.conditions}. Error: {e}", exc_info=True)
            return False

        if not conditions_schema.conditions:
            logger.warning(f"Alert {alert.id} has no conditions defined in its rule set.")
            return False

        condition_results = []
        tasks = [self._evaluate_single_condition(alert.symbol, cond_item) for cond_item in conditions_schema.conditions]
        try:
            results_from_gather = await asyncio.gather(*tasks, return_exceptions=True)
            for res in results_from_gather:
                if isinstance(res, Exception):
                    logger.error(f"Exception during single condition evaluation for alert {alert.id}: {res}", exc_info=res)
                    condition_results.append(False)
                else:
                    condition_results.append(res)
        except Exception as e:
            logger.error(f"General error in asyncio.gather for alert {alert.id} conditions: {e}", exc_info=True)
            return False

        if not condition_results or len(condition_results) != len(conditions_schema.conditions):
            logger.warning(f"Mismatch in condition results for alert {alert.id}. Expected {len(conditions_schema.conditions)}, got {len(condition_results)}.")
            return False

        if conditions_schema.logical_operator.upper() == "AND":
            return all(condition_results)
        elif conditions_schema.logical_operator.upper() == "OR":
            return any(condition_results)

        logger.warning(f"Unsupported logical operator '{conditions_schema.logical_operator}' for alert {alert.id}. Defaulting to AND.")
        return all(condition_results)

    async def check_and_trigger_alerts(self):
        """
        Iterates through active alerts, evaluates them, and triggers notifications.
        This would be called periodically by a background task.
        """
        active_alerts_stmt = select(AlertModel).where(AlertModel.is_active == True)
        result = await self.db.execute(active_alerts_stmt)
        active_alerts = result.scalars().all()

        logger.info(f"Checking {len(active_alerts)} active alerts.")

        alerts_to_update_in_db = []
        for alert in active_alerts:
            try:
                should_trigger = await self._evaluate_alert_conditions(alert)
                if should_trigger:
                    logger.info(f"Alert {alert.id} for user {alert.user_id} on symbol {alert.symbol} TRIGGERED.")
                    alert.last_triggered_at = datetime.utcnow()
                    alerts_to_update_in_db.append(alert)
            except Exception as e:
                logger.error(f"Error processing alert {alert.id} during check_and_trigger_alerts: {e}", exc_info=True)

        if alerts_to_update_in_db:
            for alert_to_update in alerts_to_update_in_db:
                # Send WebSocket notification
                notification = {
                    "type": "alert_triggered",
                    "title": f"Alert Triggered: {alert_to_update.name or alert_to_update.symbol}",
                    "description": f"Your alert for {alert_to_update.symbol} has been triggered.",
                    "timestamp": datetime.utcnow().isoformat(),
                    "details": {
                        "alert_id": alert_to_update.id,
                        "symbol": alert_to_update.symbol,
                    },
                }
                await websocket_service_instance.send_personal_message(
                    notification, str(alert_to_update.user_id)
                )
                self.db.add(alert_to_update)
            try:
                await self.db.commit()
                logger.info(f"Successfully updated {len(alerts_to_update_in_db)} triggered alerts in DB.")
            except Exception as e:
                logger.error(f"Failed to commit updates for triggered alerts: {e}", exc_info=True)
                await self.db.rollback()

# Dependency for AlertService
async def get_alert_service(
    db: AsyncSession = Depends(get_db),
    sentiment_service: SentimentAnalysisService = Depends(get_sentiment_service),
    volatility_service: VolatilityService = Depends(get_volatility_service)
) -> AlertService:
    return AlertService(
        db=db,
        sentiment_service=sentiment_service,
        volatility_service=volatility_service,
    )
