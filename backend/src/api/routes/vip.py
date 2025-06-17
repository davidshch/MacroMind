from fastapi import APIRouter, HTTPException, Depends, WebSocket, WebSocketDisconnect, Query, Body, status
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from datetime import date, datetime

from ...services.ai_explanation import AIExplanationService, get_ai_explanation_service
from ...schemas.ai_explanation import AIExplanationRequest, AIExplanationResponse
from ...services.auth import get_current_user, get_current_active_user
from ...database.models import User as UserModel
from ...database.database import get_db
from ...services.websocket import websocket_service_instance

router = APIRouter(prefix="/api/v1", tags=["VIP Features"])

# --- AI Explanation Endpoint (New) --- #
@router.post(
    "/ai/explain", 
    response_model=AIExplanationResponse, 
    tags=["AI Explanation", "VIP"],
    summary="Get AI-Powered Explanation",
    description="Provides an AI-generated explanation for a given query and context. Access may be restricted."
)
async def get_ai_powered_explanation(
    request_data: AIExplanationRequest,
    current_user: UserModel = Depends(get_current_active_user),
    ai_service: AIExplanationService = Depends(get_ai_explanation_service)
):
    """(VIP) Get AI-powered explanation based on a query and context."""
    if not current_user.is_vip:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access to AI explanations is restricted to VIP users.")
    
    try:
        explanation_response = await ai_service.generate_explanation(request_data)
        return explanation_response
    except HTTPException as he:
        raise he
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.exception(f"Error in /ai/explain for query '{request_data.query}': {e}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="An unexpected error occurred while generating the AI explanation.")

# --- WebSocket Endpoint --- #
@router.websocket("/ws/vip/{user_id}")
async def vip_websocket_endpoint(websocket: WebSocket, user_id: str):
    await websocket_service_instance.connect(websocket, user_id)
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"VIP Message received: {data} by {user_id}")
    except WebSocketDisconnect:
        websocket_service_instance.disconnect(websocket, user_id)
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"VIP WebSocket disconnected for user {user_id}")
    except Exception as e:
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"VIP WebSocket error for user {user_id}: {e}")
        websocket_service_instance.disconnect(websocket, user_id)
