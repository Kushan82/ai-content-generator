from fastapi import Depends, HTTPException
from typing import Dict, Any, Optional
import asyncio

from app.services.workflow_engine import WorkflowEngine
from app.api.websockets.workflow_updates import WebSocketManager
from app.core.config import settings
from app.core.logging import logger

# Global variables to store instances (will be set by main.py)
_workflow_engine: Optional[WorkflowEngine] = None
_websocket_manager: Optional[WebSocketManager] = None
_agents: Optional[Dict[str, Any]] = None

def set_workflow_engine(workflow_engine: WorkflowEngine):
    """Set the workflow engine instance (called from main.py during startup)"""
    global _workflow_engine
    _workflow_engine = workflow_engine

def set_websocket_manager(websocket_manager: WebSocketManager):
    """Set the WebSocket manager instance (called from main.py during startup)"""
    global _websocket_manager
    _websocket_manager = websocket_manager

def set_agents(agents: Dict[str, Any]):
    """Set the agents dictionary (called from main.py during startup)"""
    global _agents
    _agents = agents

async def get_workflow_engine() -> WorkflowEngine:
    """Dependency to get workflow engine instance"""
    if _workflow_engine is None:
        raise HTTPException(
            status_code=503, 
            detail="Workflow engine not initialized. Please check if the application started correctly."
        )
    return _workflow_engine

async def get_websocket_manager() -> WebSocketManager:
    """Dependency to get WebSocket manager instance"""
    if _websocket_manager is None:
        raise HTTPException(
            status_code=503,
            detail="WebSocket manager not initialized. Please check if the application started correctly."
        )
    return _websocket_manager

async def get_agents() -> Dict[str, Any]:
    """Dependency to get agents dictionary"""
    if _agents is None:
        raise HTTPException(
            status_code=503,
            detail="Agents not initialized. Please check if the application started correctly."
        )
    return _agents

async def get_agent_by_id(agent_id: str) -> Any:
    """Get a specific agent by ID"""
    agents = await get_agents()
    if agent_id not in agents:
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_id}' not found. Available agents: {list(agents.keys())}"
        )
    return agents[agent_id]

# Health check dependency
async def verify_system_health():
    """Verify that all critical system components are healthy"""
    try:
        # Check workflow engine
        workflow_engine = await get_workflow_engine()
        
        # Check WebSocket manager
        websocket_manager = await get_websocket_manager()
        
        # Check agents
        agents = await get_agents()
        
        return {
            "workflow_engine": "healthy",
            "websocket_manager": "healthy", 
            "agents_count": len(agents),
            "status": "all_systems_operational"
        }
    
    except Exception as e:
        logger.error(f"System health check failed: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"System health check failed: {str(e)}"
        )

# Rate limiting dependency (placeholder for future implementation)
async def rate_limit_check():
    """Check rate limits for API requests"""
    # Implementation would go here
    pass

# Authentication dependency (placeholder for future implementation)
async def get_current_user():
    """Get current authenticated user (placeholder)"""
    # Implementation would go here
    return {"user_id": "anonymous", "permissions": ["read", "write"]}
