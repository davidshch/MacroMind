"""
API routes for managing user alerts.
"""
import logging
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Query

from src.schemas.alert import AlertCreate, AlertResponse, AlertUpdate
from src.services.alerts import AlertService, get_alert_service
from src.database.models import User as UserModel # For current_user dependency
from src.services.auth import get_current_active_user # For user authentication

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post(
    "/", 
    response_model=AlertResponse, 
    status_code=status.HTTP_201_CREATED,
    summary="Create a new alert",
    description="Allows an authenticated user to create a new personalized alert."
)
async def create_alert(
    alert_data: AlertCreate,
    alert_service: AlertService = Depends(get_alert_service),
    current_user: UserModel = Depends(get_current_active_user)
) -> AlertResponse:
    """
    Creates a new alert for the authenticated user.
    """
    try:
        created_alert = await alert_service.create_alert(user_id=current_user.id, alert_data=alert_data)
        return AlertResponse.model_validate(created_alert)
    except HTTPException as http_exc:
        logger.error(f"HTTPException during alert creation for user {current_user.id}: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.exception(f"Unexpected error creating alert for user {current_user.id}, data: {alert_data.model_dump_json()}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while creating alert.")

@router.get(
    "/{alert_id}", 
    response_model=AlertResponse,
    summary="Get a specific alert",
    description="Retrieves a specific alert by its ID, if it belongs to the authenticated user."
)
async def get_alert(
    alert_id: int,
    alert_service: AlertService = Depends(get_alert_service),
    current_user: UserModel = Depends(get_current_active_user)
) -> AlertResponse:
    """
    Retrieves a specific alert by ID for the authenticated user.
    """
    alert = await alert_service.get_alert_by_id(alert_id=alert_id, user_id=current_user.id)
    if not alert:
        logger.warning(f"Alert {alert_id} not found for user {current_user.id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
    return AlertResponse.model_validate(alert)

@router.get(
    "/", 
    response_model=List[AlertResponse],
    summary="List all alerts for the user",
    description="Retrieves all alerts configured by the authenticated user."
)
async def list_user_alerts(
    skip: int = Query(0, ge=0, description="Number of records to skip for pagination"),
    limit: int = Query(100, ge=1, le=200, description="Maximum number of records to return"),
    alert_service: AlertService = Depends(get_alert_service),
    current_user: UserModel = Depends(get_current_active_user)
) -> List[AlertResponse]:
    """
    Lists all alerts for the authenticated user with pagination.
    """
    alerts = await alert_service.get_user_alerts(user_id=current_user.id, skip=skip, limit=limit)
    return [AlertResponse.model_validate(alert) for alert in alerts]

@router.put(
    "/{alert_id}", 
    response_model=AlertResponse,
    summary="Update an existing alert",
    description="Allows an authenticated user to update their existing alert by ID."
)
async def update_alert(
    alert_id: int,
    alert_update_data: AlertUpdate,
    alert_service: AlertService = Depends(get_alert_service),
    current_user: UserModel = Depends(get_current_active_user)
) -> AlertResponse:
    """
    Updates an existing alert for the authenticated user.
    """
    try:
        updated_alert = await alert_service.update_alert(
            alert_id=alert_id, user_id=current_user.id, alert_update_data=alert_update_data
        )
        if not updated_alert:
            logger.warning(f"Alert {alert_id} not found for update by user {current_user.id}")
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found or no update performed")
        return AlertResponse.model_validate(updated_alert)
    except HTTPException as http_exc:
        # Re-raise known HTTPExceptions (e.g., from service layer for bad data)
        raise http_exc
    except Exception as e:
        logger.exception(f"Unexpected error updating alert {alert_id} for user {current_user.id}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error while updating alert.")


@router.delete(
    "/{alert_id}", 
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete an alert",
    description="Allows an authenticated user to delete their alert by ID."
)
async def delete_alert(
    alert_id: int,
    alert_service: AlertService = Depends(get_alert_service),
    current_user: UserModel = Depends(get_current_active_user)
):
    """
    Deletes an alert for the authenticated user.
    """
    deleted = await alert_service.delete_alert(alert_id=alert_id, user_id=current_user.id)
    if not deleted:
        logger.warning(f"Alert {alert_id} not found for deletion by user {current_user.id}")
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Alert not found")
    return None # HTTP 204 No Content
