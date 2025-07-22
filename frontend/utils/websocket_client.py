import asyncio
import json
import websockets
from typing import Dict, Any, Optional, Callable
import threading
import time
from datetime import datetime

class WebSocketClient:
    """
    WebSocket client for real-time communication with the FastAPI backend.
    Handles connection management, message handling, and automatic reconnection.
    """
    
    def __init__(self, url: str = "ws://localhost:8000/ws/workflow"):
        self.url = url
        self.websocket = None
        self.is_connected = False
        self.is_connecting = False
        self.message_handlers = {}
        self.connection_callbacks = []
        self.disconnect_callbacks = []
        
        # Threading for async operations in Streamlit
        self.loop = None
        self.thread = None
        
    def add_message_handler(self, message_type: str, handler: Callable):
        """Add a message handler for specific message types."""
        self.message_handlers[message_type] = handler
    
    def add_connection_callback(self, callback: Callable):
        """Add callback for connection events."""
        self.connection_callbacks.append(callback)
    
    def add_disconnect_callback(self, callback: Callable):
        """Add callback for disconnection events."""
        self.disconnect_callbacks.append(callback)
    
    async def connect(self):
        """Connect to WebSocket server."""
        if self.is_connected or self.is_connecting:
            return
        
        self.is_connecting = True
        
        try:
            self.websocket = await websockets.connect(self.url)
            self.is_connected = True
            self.is_connecting = False
            
            # Notify connection callbacks
            for callback in self.connection_callbacks:
                callback()
            
            # Start message listening
            await self.listen_for_messages()
            
        except Exception as e:
            self.is_connecting = False
            print(f"WebSocket connection failed: {e}")
    
    async def disconnect(self):
        """Disconnect from WebSocket server."""
        if self.websocket and self.is_connected:
            await self.websocket.close()
            self.is_connected = False
            
            # Notify disconnect callbacks
            for callback in self.disconnect_callbacks:
                callback()
    
    async def send_message(self, message: Dict[str, Any]):
        """Send message to WebSocket server."""
        if not self.is_connected or not self.websocket:
            print("WebSocket not connected")
            return
        
        try:
            message_json = json.dumps(message)
            await self.websocket.send(message_json)
        except Exception as e:
            print(f"Failed to send WebSocket message: {e}")
    
    async def listen_for_messages(self):
        """Listen for incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    message_type = data.get("type", "unknown")
                    
                    # Handle message with registered handler
                    if message_type in self.message_handlers:
                        self.message_handlers[message_type](data)
                    else:
                        print(f"Unhandled message type: {message_type}")
                        
                except json.JSONDecodeError:
                    print(f"Invalid JSON message: {message}")
                    
        except websockets.exceptions.ConnectionClosed:
            self.is_connected = False
            print("WebSocket connection closed")
            
        except Exception as e:
            self.is_connected = False
            print(f"WebSocket listening error: {e}")
    
    def subscribe_to_workflow(self, workflow_id: str):
        """Subscribe to workflow updates."""
        if self.is_connected:
            asyncio.create_task(self.send_message({
                "type": "subscribe_workflow",
                "workflow_id": workflow_id
            }))
    
    def get_agent_status(self):
        """Request agent status update."""
        if self.is_connected:
            asyncio.create_task(self.send_message({
                "type": "get_agent_status"
            }))
    
    def ping(self):
        """Send ping to keep connection alive."""
        if self.is_connected:
            asyncio.create_task(self.send_message({
                "type": "ping",
                "timestamp": datetime.now().isoformat()
            }))
