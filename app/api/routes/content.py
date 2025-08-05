import asyncio
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import json
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, WebSocket
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, validator

from app.models.content_schemas import ContentRequest, ContentResponse, Persona
from app.models.workflow_schemas import WorkflowState
from app.agents.orchestrator_agent import OrchestratorAgent
from app.services.workflow_engine import WorkflowEngine
from app.core.config import settings
from app.core.logging import logger
from app.utils.dependencies import get_workflow_engine, get_websocket_manager

router = APIRouter(prefix="/content", tags=["content"])

class MultiAgentContentRequest(BaseModel):
    """
    Enhanced content request for multi-agent generation with comprehensive configuration options.
    """
    # Core content parameters
    persona_id: str = Field(..., description="Target persona identifier")
    content_type: str = Field(..., description="Type of content to generate")
    topic: str = Field(..., min_length=5, max_length=200, description="Content topic or subject")
    
    # Content customization
    word_count: int = Field(default=200, ge=50, le=2000, description="Target word count")
    creativity_level: float = Field(default=0.7, ge=0.0, le=1.0, description="Creativity level (0-1)")
    urgency_level: str = Field(default="medium", description="Urgency level: low, medium, high")
    
    # Advanced configuration
    additional_context: Optional[str] = Field(None, max_length=1000, description="Additional context or requirements")
    brand_voice: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Brand voice guidelines")
    business_objectives: Optional[list] = Field(default_factory=list, description="Business objectives")
    
    # Workflow configuration
    workflow_type: str = Field(default="standard", description="Workflow type: standard, rapid, research_heavy")
    enable_qa: bool = Field(default=True, description="Enable quality assurance step")
    enable_variations: bool = Field(default=False, description="Generate content variations")
    
    # Real-time updates
    enable_realtime_updates: bool = Field(default=False, description="Enable WebSocket real-time updates")
    
    # Request metadata
    request_id: Optional[str] = Field(default_factory=lambda: str(uuid.uuid4()))
    client_info: Optional[Dict[str, str]] = Field(default_factory=dict)
    
    @validator('content_type')
    def validate_content_type(cls, v):
        allowed_types = ['ad', 'landing_page', 'blog_intro', 'email', 'social_media', 'product_description']
        if v not in allowed_types:
            raise ValueError(f'Content type must be one of: {", ".join(allowed_types)}')
        return v

class ContentGenerationResponse(BaseModel):
    """
    Comprehensive response from multi-agent content generation workflow.
    """
    # Primary content output
    final_content: str = Field(..., description="Final optimized content")
    content_variations: list = Field(default_factory=list, description="Alternative content versions")
    
    # Workflow execution details
    workflow_summary: Dict[str, Any] = Field(..., description="Workflow execution summary")
    agent_contributions: Dict[str, Any] = Field(..., description="Individual agent outputs")
    
    # Quality and performance metrics
    quality_assessment: Dict[str, Any] = Field(default_factory=dict, description="Quality evaluation results")
    performance_metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance statistics")
    
    # Metadata and insights
    generation_metadata: Dict[str, Any] = Field(default_factory=dict, description="Generation metadata")
    optimization_suggestions: list = Field(default_factory=list, description="Improvement recommendations")
    
    # Request tracking
    request_id: str = Field(..., description="Original request identifier")
    workflow_id: str = Field(..., description="Workflow execution identifier")
    timestamp: datetime = Field(default_factory=datetime.utcnow)

@router.post("/generate", response_model=ContentGenerationResponse)
async def generate_content_multi_agent(
    request: MultiAgentContentRequest,
    background_tasks: BackgroundTasks,
    workflow_engine: WorkflowEngine = Depends(get_workflow_engine),
    websocket_manager = Depends(get_websocket_manager)
):
    """
    Generate content using the complete multi-agent workflow.
    
    This endpoint orchestrates multiple specialized AI agents to create
    high-quality, persona-tailored marketing content with comprehensive
    quality assurance and optimization.
    """
    logger.info(
        "Multi-agent content generation request received",
        request_id=request.request_id,
        content_type=request.content_type,
        topic=request.topic,
        persona_id=request.persona_id
    )
    
    try:
        # Prepare workflow request
        workflow_request = {
            "request_id": request.request_id,
            "persona_id": request.persona_id,
            "content_type": request.content_type,
            "topic": request.topic,
            "word_count": request.word_count,
            "creativity_level": request.creativity_level,
            "additional_context": request.additional_context,
            "brand_voice": request.brand_voice,
            "business_objectives": request.business_objectives,
            "skip_qa": not request.enable_qa,
            "generate_variations": request.enable_variations
        }
        
        # Execute multi-agent workflow
        if request.enable_realtime_updates:
            # Background execution with WebSocket updates
            background_tasks.add_task(
                execute_workflow_with_updates,
                workflow_engine,
                workflow_request,
                websocket_manager
            )
            
            return ContentGenerationResponse(
                final_content="Generation in progress...",
                workflow_summary={"status": "started", "realtime_updates": True},
                agent_contributions={},
                request_id=request.request_id,
                workflow_id="pending"
            )
        
        else:
            # Synchronous execution
            workflow_result = await workflow_engine.execute_workflow(workflow_request)
            
            # Transform workflow result to API response
            response = transform_workflow_result_to_response(workflow_result, request.request_id)
            
            logger.info(
                "Multi-agent content generation completed",
                request_id=request.request_id,
                workflow_id=response.workflow_id,
                quality_score=response.quality_assessment.get("overall_score", 0),
                content_length=len(response.final_content)
            )
            
            return response
    
    except Exception as e:
        logger.error(
            "Multi-agent content generation failed",
            request_id=request.request_id,
            error=str(e)
        )
        
        raise HTTPException(
            status_code=500,
            detail={
                "message": "Content generation failed",
                "error": str(e),
                "request_id": request.request_id
            }
        )

@router.get("/generate/stream/{workflow_id}")
async def stream_content_generation(workflow_id: str):
    """
    Stream real-time updates from a content generation workflow.
    
    Provides server-sent events (SSE) for real-time workflow monitoring
    without requiring WebSocket connections.
    """
    async def generate_workflow_stream():
        """Generate SSE stream for workflow updates."""
        workflow_engine = get_workflow_engine()
        
        while True:
            try:
                # Get current workflow status
                status = await workflow_engine.get_workflow_status(workflow_id)
                
                if not status:
                    yield f"data: {json.dumps({'error': 'Workflow not found'})}\n\n"
                    break
                
                # Send status update
                yield f"data: {json.dumps(status)}\n\n"
                
                # Check if workflow completed
                if status.get("status") in ["completed", "failed", "cancelled"]:
                    break
                
                # Wait before next update
                await asyncio.sleep(2)
                
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                break
    
    return StreamingResponse(
        generate_workflow_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )

@router.get("/personas", response_model=List[Persona])
async def get_available_personas():
    """
    Get all available persona profiles for content generation.
    
    Returns detailed information about each persona including demographics,
    psychographics, pain points, and communication preferences.
    """
    try:
        # Import persona service to get available personas
        from app.agents.persona_research_agent import PersonaResearchAgent
        from app.services.llm_service import llm_service
        
        # Create temporary persona research agent to get persona data
        persona_agent = PersonaResearchAgent(llm_service)
        personas = persona_agent.persona_service.list_personas()

        return personas
    
    except Exception as e:
        logger.error("Failed to retrieve personas", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve personas")

@router.get("/types")
async def get_content_types():
    """
    Get all supported content types with descriptions and specifications.
    
    Returns information about supported content formats including
    optimal word counts, structure requirements, and use cases.
    """
    content_types = [
        {
            "id": "ad",
            "name": "Advertisement",
            "description": "Short-form marketing advertisement copy",
            "optimal_word_count": "25-75 words",
            "structure": "Hook → Value Prop → CTA",
            "use_cases": ["Social media ads", "Display advertising", "PPC campaigns"]
        },
        {
            "id": "landing_page", 
            "name": "Landing Page",
            "description": "Conversion-focused landing page copy",
            "optimal_word_count": "150-400 words",
            "structure": "Headline → Benefits → Social Proof → CTA",
            "use_cases": ["Lead generation", "Product launches", "Campaign pages"]
        },
        {
            "id": "blog_intro",
            "name": "Blog Introduction", 
            "description": "Engaging blog post introduction",
            "optimal_word_count": "100-200 words",
            "structure": "Hook → Problem → Promise → Preview",
            "use_cases": ["Content marketing", "SEO articles", "Thought leadership"]
        },
        {
            "id": "email",
            "name": "Email Campaign",
            "description": "Email marketing content",
            "optimal_word_count": "75-250 words", 
            "structure": "Subject → Greeting → Value → CTA",
            "use_cases": ["Newsletter", "Promotional emails", "Nurture sequences"]
        },
        {
            "id": "social_media",
            "name": "Social Media Post",
            "description": "Platform-optimized social content",
            "optimal_word_count": "Platform-dependent",
            "structure": "Hook → Message → Engagement → CTA",
            "use_cases": ["Brand awareness", "Community engagement", "Lead generation"]
        }
    ]
    
    return {"content_types": content_types}

@router.get("/status/{request_id}")
async def get_content_generation_status(
    request_id: str,
    workflow_engine: WorkflowEngine = Depends(get_workflow_engine)
):
    """
    Get the current status of a content generation request.
    
    Provides detailed information about workflow progress,
    agent status, and estimated completion time.
    """
    try:
        # Find workflow by request_id (this would need to be implemented)
        # For now, return mock status
        return {
            "request_id": request_id,
            "status": "in_progress",
            "current_step": "content_generation",
            "progress_percentage": 65,
            "estimated_completion": "2 minutes",
            "agents_completed": ["persona_researcher", "content_strategist"],
            "current_agent": "creative_generator"
        }
    
    except Exception as e:
        logger.error("Failed to get generation status", request_id=request_id, error=str(e))
        raise HTTPException(status_code=404, detail="Request not found")

# Helper functions
async def execute_workflow_with_updates(
    workflow_engine: WorkflowEngine,
    workflow_request: Dict[str, Any], 
    websocket_manager
):
    """Execute workflow with real-time WebSocket updates."""
    try:
        workflow_result = await workflow_engine.execute_workflow(workflow_request)
        
        # Send completion notification via WebSocket
        await websocket_manager.broadcast({
            "type": "workflow_completed",
            "request_id": workflow_request["request_id"],
            "workflow_id": workflow_result.workflow_id,
            "final_content": workflow_result.final_content,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        # Send error notification via WebSocket
        await websocket_manager.broadcast({
            "type": "workflow_error",
            "request_id": workflow_request["request_id"],
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        })

def transform_workflow_result_to_response(
    workflow_result: WorkflowState,
    request_id: str
) -> ContentGenerationResponse:
    """Transform workflow result to API response format."""
    
    # Extract quality assessment if available
    quality_assessment = {}
    if "quality_assurance" in workflow_result.agent_responses:
        qa_response = workflow_result.agent_responses["quality_assurance"]
        if hasattr(qa_response, 'content'):
            quality_assessment = qa_response.content.get("quality_assessment", {})
        elif isinstance(qa_response, dict):
            quality_assessment = qa_response.get("content", {}).get("quality_assessment", {})
    
    # Create comprehensive response
    return ContentGenerationResponse(
        final_content=workflow_result.final_content or "Content generation failed",
        content_variations=[],  # Would extract variations if available
        workflow_summary={
            "workflow_id": workflow_result.workflow_id,
            "status": workflow_result.status.value if hasattr(workflow_result.status, 'value') else workflow_result.status,
            "agents_involved": len(workflow_result.agent_responses),
            "total_duration": workflow_result.total_duration_seconds,
            "success": len(workflow_result.errors) == 0
        },
        agent_contributions={
            agent_id: response.dict() if hasattr(response, 'dict') else response
            for agent_id, response in workflow_result.agent_responses.items()
        },
        quality_assessment=quality_assessment,
        performance_metrics={
            "overall_confidence": workflow_result.overall_confidence_score,
            "total_tokens": workflow_result.total_tokens_used,
            "total_api_calls": workflow_result.total_api_calls,
            "processing_time": workflow_result.total_duration_seconds
        },
        generation_metadata={
            "workflow_type": workflow_result.workflow_type,
            "agents_count": len(workflow_result.agent_responses),
            "retry_count": workflow_result.retry_count
        },
        optimization_suggestions=[],  # Would extract from QA results
        request_id=request_id,
        workflow_id=workflow_result.workflow_id
    )
