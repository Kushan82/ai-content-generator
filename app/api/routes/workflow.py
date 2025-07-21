from fastapi import APIRouter, Depends
from app.utils.dependencies import get_workflow_engine

router = APIRouter(prefix="/workflow", tags=["workflow"])

@router.get("/status/{workflow_id}")
async def get_workflow_status(workflow_id: str, workflow_engine = Depends(get_workflow_engine)):
    """Get workflow status"""
    return await workflow_engine.get_workflow_status(workflow_id)
