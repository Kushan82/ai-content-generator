from fastapi import APIRouter, Depends
from app.utils.dependencies import get_agents, get_agent_by_id

router = APIRouter(prefix="/agents", tags=["agents"])

@router.get("/status")
async def get_all_agents_status(agents = Depends(get_agents)):
    """Get status of all agents"""
    agent_status = {}
    for agent_id, agent in agents.items():
        agent_status[agent_id] = agent.get_agent_info()
    return agent_status

@router.get("/{agent_id}/status") 
async def get_agent_status(agent_id: str, agent = Depends(get_agent_by_id)):
    """Get status of specific agent"""
    return agent.get_agent_info()
