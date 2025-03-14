from typing import Dict, List, Any
from fastapi import WebSocket
import json
import asyncio
import logging
from datetime import datetime
from .market_data import MarketDataService
from .sentiment_analysis import SentimentAnalysisService
from .alerts import AlertService

logger = logging.getLogger(__name__)

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.market_service = MarketDataService()
        self.sentiment_service = SentimentAnalysisService()

    async def connect(self, websocket: WebSocket, symbol: str):
        await websocket.accept()
        if symbol not in self.active_connections:
            self.active_connections[symbol] = []
        self.active_connections[symbol].append(websocket)
        logger.info(f"New connection for {symbol}. Total connections: {len(self.active_connections[symbol])}")

    def disconnect(self, websocket: WebSocket, symbol: str):
        self.active_connections[symbol].remove(websocket)
        if not self.active_connections[symbol]:
            del self.active_connections[symbol]
        logger.info(f"Connection closed for {symbol}")

    async def broadcast_to_symbol(self, symbol: str, message: Dict[str, Any]):
        if symbol not in self.active_connections:
            return
        
        for connection in self.active_connections[symbol]:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {symbol}: {str(e)}")

    async def start_streaming(self):
        """Start streaming market data for all connected symbols."""
        while True:
            try:
                for symbol in list(self.active_connections.keys()):
                    data = await self._get_real_time_data(symbol)
                    await self.broadcast_to_symbol(symbol, data)
            except Exception as e:
                logger.error(f"Streaming error: {str(e)}")
            
            await asyncio.sleep(5)  # Update every 5 seconds

    async def _get_real_time_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time market data and sentiment."""
        try:
            market_data = await self.market_service.get_stock_data(symbol)
            sentiment = await self.sentiment_service.get_market_sentiment(symbol)
            
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "price": market_data.get("price"),
                "change": market_data.get("change"),
                "sentiment": sentiment.get("sentiment"),
                "sentiment_confidence": sentiment.get("confidence")
            }
        except Exception as e:
            logger.error(f"Error getting real-time data for {symbol}: {str(e)}")
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
