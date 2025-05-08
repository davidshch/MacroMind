"""WebSocket connection management and broadcasting."""

import logging
from typing import List, Dict, Any, Optional
from fastapi import WebSocket, WebSocketDisconnect
import json

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manages active WebSocket connections."""
    def __init__(self):
        # Store connections mapped by user_id (assuming user_id is available)
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, user_id: str):
        """Accept a new WebSocket connection and store it."""
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
        logger.info(f"WebSocket connected for user {user_id}. Total connections for user: {len(self.active_connections[user_id])}")

    def disconnect(self, websocket: WebSocket, user_id: str):
        """Remove a WebSocket connection."""
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
                if not self.active_connections[user_id]: # Remove user entry if no connections left
                    del self.active_connections[user_id]
                logger.info(f"WebSocket disconnected for user {user_id}.")
            else:
                 logger.warning(f"Attempted to disconnect unknown websocket for user {user_id}.")
        else:
            logger.warning(f"Attempted to disconnect websocket for unknown user {user_id}.")

    async def send_personal_message(self, message: Any, user_id: str):
        """Send a message to all connections for a specific user."""
        if user_id in self.active_connections:
            connections = self.active_connections[user_id]
            message_json = json.dumps(message) # Serialize message once
            logger.debug(f"Sending message to user {user_id} ({len(connections)} connections): {message_json}")
            # Use gather for concurrent sending
            results = await asyncio.gather(
                *[conn.send_text(message_json) for conn in connections],
                return_exceptions=True
            )
            # Log any errors during sending
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to send message to user {user_id} connection {i}: {result}")
                    # Optionally handle disconnects here if send fails
                    # self.disconnect(connections[i], user_id)
        else:
            logger.debug(f"No active WebSocket connections found for user {user_id} to send message.")

    async def broadcast(self, message: Any):
        """Send a message to all connected clients."""
        message_json = json.dumps(message)
        logger.debug(f"Broadcasting message to all users: {message_json}")
        all_connections = [conn for user_conns in self.active_connections.values() for conn in user_conns]
        if all_connections:
            results = await asyncio.gather(
                *[conn.send_text(message_json) for conn in all_connections],
                return_exceptions=True
            )
            # Log any errors during broadcast
            for i, result in enumerate(results):
                 if isinstance(result, Exception):
                     logger.error(f"Failed to broadcast message to connection {i}: {result}")

# --- WebSocket Service (Singleton Pattern) --- #
# This makes it easy to access the manager from other services like AlertsService

class WebSocketService:
    _instance: Optional[ConnectionManager] = None

    @classmethod
    def get_manager(cls) -> ConnectionManager:
        """Get the singleton instance of the ConnectionManager."""
        if cls._instance is None:
            cls._instance = ConnectionManager()
            logger.info("Initialized WebSocket ConnectionManager singleton.")
        return cls._instance

    async def connect(self, websocket: WebSocket, user_id: str):
        await self.get_manager().connect(websocket, user_id)

    def disconnect(self, websocket: WebSocket, user_id: str):
        self.get_manager().disconnect(websocket, user_id)

    async def send_personal_message(self, message: Any, user_id: str):
        await self.get_manager().send_personal_message(message, user_id)

    async def broadcast(self, message: Any):
        await self.get_manager().broadcast(message)

# Instantiate the service to make get_manager available
# This approach might be debated vs. using FastAPI Depends, but works for service-to-service calls.
websocket_service_instance = WebSocketService()

# You can then import and use `websocket_service_instance` in other services
# e.g., from .websocket import websocket_service_instance
# await websocket_service_instance.send_personal_message(...)
