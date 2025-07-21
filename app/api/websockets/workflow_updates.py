import asyncio
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from app.core.logging import logger

class WebSocketManager:
    """
    WebSocket connection manager for real-time workflow monitoring and agent communication.
    
    Manages multiple WebSocket connections, provides broadcasting capabilities,
    and handles workflow-specific subscriptions for targeted updates.
    """
    
    def __init__(self):
        # Active WebSocket connections
        self.active_connections: List[WebSocket] = []
        
        # Connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        
        # Workflow subscriptions - maps workflow_id to list of WebSocket connections
        self.workflow_subscriptions: Dict[str, List[WebSocket]] = {}
        
        # Agent status subscribers
        self.agent_subscribers: List[WebSocket] = []
        
        # Connection statistics
        self.connection_stats = {
            "total_connections": 0,
            "active_connections": 0,
            "messages_sent": 0,
            "errors": 0
        }
        
        logger.info("WebSocket Manager initialized")
    
    async def connect(self, websocket: WebSocket, client_info: Optional[Dict[str, Any]] = None):
        """
        Accept a new WebSocket connection and initialize client session.
        """
        try:
            await websocket.accept()
            
            # Add to active connections
            self.active_connections.append(websocket)
            
            # Store connection metadata
            self.connection_metadata[websocket] = {
                "connected_at": datetime.utcnow(),
                "client_info": client_info or {},
                "subscriptions": [],
                "messages_sent": 0,
                "client_id": str(uuid.uuid4())[:8]
            }
            
            # Update statistics
            self.connection_stats["total_connections"] += 1
            self.connection_stats["active_connections"] = len(self.active_connections)
            
            client_id = self.connection_metadata[websocket]["client_id"]
            logger.info(
                "WebSocket connection established",
                client_id=client_id,
                total_active=len(self.active_connections)
            )
            
            # Send welcome message
            await self.send_personal_message(websocket, {
                "type": "connection_established",
                "client_id": client_id,
                "message": "Connected to Multi-Agent Content Generator",
                "capabilities": [
                    "workflow_monitoring",
                    "agent_status_updates", 
                    "real_time_generation_progress",
                    "error_notifications"
                ],
                "timestamp": datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            logger.error("Failed to establish WebSocket connection", error=str(e))
            self.connection_stats["errors"] += 1
            raise
    
    def disconnect(self, websocket: WebSocket):
        """
        Handle WebSocket disconnection and cleanup.
        """
        try:
            # Remove from active connections
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)
            
            # Get client info before cleanup
            client_metadata = self.connection_metadata.get(websocket, {})
            client_id = client_metadata.get("client_id", "unknown")
            
            # Remove from workflow subscriptions
            for workflow_id, subscribers in self.workflow_subscriptions.items():
                if websocket in subscribers:
                    subscribers.remove(websocket)
            
            # Remove from agent subscribers
            if websocket in self.agent_subscribers:
                self.agent_subscribers.remove(websocket)
            
            # Clean up metadata
            if websocket in self.connection_metadata:
                del self.connection_metadata[websocket]
            
            # Update statistics
            self.connection_stats["active_connections"] = len(self.active_connections)
            
            logger.info(
                "WebSocket connection closed",
                client_id=client_id,
                remaining_connections=len(self.active_connections)
            )
            
        except Exception as e:
            logger.error("Error during WebSocket disconnect cleanup", error=str(e))
    
    async def send_personal_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """
        Send a message to a specific WebSocket connection.
        """
        try:
            message_json = json.dumps(message, default=str)
            await websocket.send_text(message_json)
            
            # Update statistics
            self.connection_stats["messages_sent"] += 1
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["messages_sent"] += 1
            
        except WebSocketDisconnect:
            # Handle client disconnect
            self.disconnect(websocket)
            
        except Exception as e:
            logger.error("Failed to send personal WebSocket message", error=str(e))
            self.connection_stats["errors"] += 1
    
    async def broadcast(self, message: Dict[str, Any], exclude: Optional[List[WebSocket]] = None):
        """
        Broadcast a message to all connected WebSocket clients.
        """
        if not self.active_connections:
            return
        
        exclude = exclude or []
        message_json = json.dumps(message, default=str)
        
        # Send to all active connections (except excluded ones)
        disconnected_connections = []
        
        for websocket in self.active_connections:
            if websocket in exclude:
                continue
            
            try:
                await websocket.send_text(message_json)
                
                # Update statistics
                self.connection_stats["messages_sent"] += 1
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]["messages_sent"] += 1
                    
            except WebSocketDisconnect:
                disconnected_connections.append(websocket)
            except Exception as e:
                logger.error("Failed to broadcast to WebSocket client", error=str(e))
                disconnected_connections.append(websocket)
                self.connection_stats["errors"] += 1
        
        # Clean up disconnected connections
        for websocket in disconnected_connections:
            self.disconnect(websocket)
        
        if disconnected_connections:
            logger.info(
                "Cleaned up disconnected WebSocket connections",
                disconnected_count=len(disconnected_connections)
            )
    
    async def subscribe_to_workflow(self, websocket: WebSocket, workflow_id: str):
        """
        Subscribe a WebSocket connection to workflow-specific updates.
        """
        if workflow_id not in self.workflow_subscriptions:
            self.workflow_subscriptions[workflow_id] = []
        
        if websocket not in self.workflow_subscriptions[workflow_id]:
            self.workflow_subscriptions[workflow_id].append(websocket)
            
            # Update connection metadata
            if websocket in self.connection_metadata:
                self.connection_metadata[websocket]["subscriptions"].append(workflow_id)
            
            logger.info(
                "WebSocket subscribed to workflow",
                workflow_id=workflow_id,
                client_id=self.connection_metadata.get(websocket, {}).get("client_id", "unknown")
            )
            
            # Send confirmation
            await self.send_personal_message(websocket, {
                "type": "workflow_subscription_confirmed",
                "workflow_id": workflow_id,
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def unsubscribe_from_workflow(self, websocket: WebSocket, workflow_id: str):
        """
        Unsubscribe a WebSocket connection from workflow updates.
        """
        if workflow_id in self.workflow_subscriptions:
            if websocket in self.workflow_subscriptions[workflow_id]:
                self.workflow_subscriptions[workflow_id].remove(websocket)
                
                # Update connection metadata
                if websocket in self.connection_metadata:
                    subscriptions = self.connection_metadata[websocket]["subscriptions"]
                    if workflow_id in subscriptions:
                        subscriptions.remove(workflow_id)
                
                logger.info(
                    "WebSocket unsubscribed from workflow",
                    workflow_id=workflow_id,
                    client_id=self.connection_metadata.get(websocket, {}).get("client_id", "unknown")
                )
    
    async def broadcast_workflow_update(self, workflow_id: str, update: Dict[str, Any]):
        """
        Broadcast an update to all subscribers of a specific workflow.
        """
        if workflow_id not in self.workflow_subscriptions:
            return
        
        subscribers = self.workflow_subscriptions[workflow_id]
        if not subscribers:
            return
        
        # Add workflow context to the update
        update_with_context = {
            **update,
            "workflow_id": workflow_id,
            "update_type": "workflow_progress",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send to all workflow subscribers
        message_json = json.dumps(update_with_context, default=str)
        disconnected_connections = []
        
        for websocket in subscribers:
            try:
                await websocket.send_text(message_json)
                
                # Update statistics
                self.connection_stats["messages_sent"] += 1
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]["messages_sent"] += 1
                    
            except WebSocketDisconnect:
                disconnected_connections.append(websocket)
            except Exception as e:
                logger.error(
                    "Failed to send workflow update",
                    workflow_id=workflow_id,
                    error=str(e)
                )
                disconnected_connections.append(websocket)
                self.connection_stats["errors"] += 1
        
        # Clean up disconnected connections
        for websocket in disconnected_connections:
            self.disconnect(websocket)
        
        logger.info(
            "Workflow update broadcast completed",
            workflow_id=workflow_id,
            subscribers_reached=len(subscribers) - len(disconnected_connections),
            update_type=update.get("type", "unknown")
        )
    
    async def subscribe_to_agent_updates(self, websocket: WebSocket):
        """
        Subscribe a WebSocket connection to agent status updates.
        """
        if websocket not in self.agent_subscribers:
            self.agent_subscribers.append(websocket)
            
            logger.info(
                "WebSocket subscribed to agent updates",
                client_id=self.connection_metadata.get(websocket, {}).get("client_id", "unknown")
            )
            
            # Send confirmation
            await self.send_personal_message(websocket, {
                "type": "agent_subscription_confirmed",
                "message": "Subscribed to agent status updates",
                "timestamp": datetime.utcnow().isoformat()
            })
    
    async def broadcast_agent_update(self, agent_id: str, update: Dict[str, Any]):
        """
        Broadcast agent status updates to subscribed clients.
        """
        if not self.agent_subscribers:
            return
        
        # Add agent context to the update
        update_with_context = {
            **update,
            "agent_id": agent_id,
            "update_type": "agent_status",
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send to all agent subscribers
        message_json = json.dumps(update_with_context, default=str)
        disconnected_connections = []
        
        for websocket in self.agent_subscribers:
            try:
                await websocket.send_text(message_json)
                
                # Update statistics
                self.connection_stats["messages_sent"] += 1
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]["messages_sent"] += 1
                    
            except WebSocketDisconnect:
                disconnected_connections.append(websocket)
            except Exception as e:
                logger.error(
                    "Failed to send agent update",
                    agent_id=agent_id,
                    error=str(e)
                )
                self.connection_stats["errors"] += 1
        
        # Clean up disconnected connections
        for websocket in disconnected_connections:
            self.disconnect(websocket)
    
    async def send_workflow_progress_update(
        self,
        workflow_id: str,
        step_name: str,
        agent_id: str,
        status: str,
        progress_percentage: float,
        additional_data: Optional[Dict[str, Any]] = None
    ):
        """
        Send detailed workflow progress update to subscribers.
        """
        update = {
            "type": "workflow_step_update",
            "step_name": step_name,
            "agent_id": agent_id,
            "status": status,
            "progress_percentage": progress_percentage,
            "additional_data": additional_data or {}
        }
        
        await self.broadcast_workflow_update(workflow_id, update)
    
    async def send_workflow_completion(
        self,
        workflow_id: str,
        final_content: str,
        quality_score: float,
        performance_metrics: Dict[str, Any]
    ):
        """
        Send workflow completion notification with results.
        """
        completion_update = {
            "type": "workflow_completed",
            "final_content": final_content,
            "quality_score": quality_score,
            "performance_metrics": performance_metrics,
            "status": "completed"
        }
        
        await self.broadcast_workflow_update(workflow_id, completion_update)
    
    async def send_workflow_error(
        self,
        workflow_id: str,
        error_message: str,
        failed_step: Optional[str] = None,
        recovery_options: Optional[List[str]] = None
    ):
        """
        Send workflow error notification to subscribers.
        """
        error_update = {
            "type": "workflow_error",
            "error_message": error_message,
            "failed_step": failed_step,
            "recovery_options": recovery_options or [],
            "status": "failed"
        }
        
        await self.broadcast_workflow_update(workflow_id, error_update)
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive WebSocket connection statistics.
        """
        return {
            **self.connection_stats,
            "workflow_subscriptions": len(self.workflow_subscriptions),
            "agent_subscribers": len(self.agent_subscribers),
            "connection_details": [
                {
                    "client_id": metadata.get("client_id"),
                    "connected_at": metadata.get("connected_at").isoformat() if metadata.get("connected_at") else None,
                    "messages_sent": metadata.get("messages_sent", 0),
                    "subscriptions": len(metadata.get("subscriptions", []))
                }
                for metadata in self.connection_metadata.values()
            ]
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on WebSocket manager and connections.
        """
        # Test connection responsiveness
        healthy_connections = 0
        total_connections = len(self.active_connections)
        
        for websocket in self.active_connections.copy():  # Use copy to avoid modification during iteration
            try:
                # Send ping to test connection
                await self.send_personal_message(websocket, {
                    "type": "health_check_ping",
                    "timestamp": datetime.utcnow().isoformat()
                })
                healthy_connections += 1
                
            except Exception:
                # Connection is unhealthy, will be cleaned up automatically
                pass
        
        return {
            "status": "healthy" if healthy_connections == total_connections else "degraded",
            "total_connections": total_connections,
            "healthy_connections": healthy_connections,
            "unhealthy_connections": total_connections - healthy_connections,
            "statistics": self.connection_stats
        }
