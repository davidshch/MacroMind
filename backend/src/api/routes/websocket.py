from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from ...services.websocket import ConnectionManager
from ...services.auth import get_current_user
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)
router = APIRouter(tags=["websocket"])

manager = ConnectionManager()

@router.websocket("/ws/{symbol}")
async def websocket_endpoint(websocket: WebSocket, symbol: str):
    """WebSocket endpoint for real-time market data."""
    try:
        await manager.connect(websocket, symbol)
        try:
            while True:
                # Wait for any client messages (like subscribe/unsubscribe)
                data = await websocket.receive_text()
                # You can handle client messages here if needed
        except WebSocketDisconnect:
            manager.disconnect(websocket, symbol)
    except Exception as e:
        logger.error(f"WebSocket error: {str(e)}")
        if websocket in manager.active_connections.get(symbol, []):
            manager.disconnect(websocket, symbol)
