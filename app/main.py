import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any
import uvicorn
import datetime
import uuid
from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings, validate_config
from app.core.logging import logger, setup_logging
from app.services.llm_service import llm_service
from app.services.workflow_engine import WorkflowEngine
from app.api.websockets.workflow_updates import WebSocketManager

# Import dependencies module and set global instances
from app.utils import dependencies

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan manager for startup and shutdown operations.
    Initializes all agents, validates configuration, and sets up monitoring.
    """
    logger.info("🚀 Starting Multi-Agent AI Content Generator")
    
    try:
        # Validate configuration at startup
        validate_config()
        
        # Initialize LLM service and verify connectivity
        health_status = await llm_service.health_check()
        if not any(health_status.values()):
            raise Exception("No LLM providers are available")
        
        logger.info("✅ LLM services initialized", providers=list(health_status.keys()))
        
        # Initialize specialized agents
        agents = await initialize_agents()
        logger.info("✅ Multi-agent system initialized", agent_count=len(agents))
        
        # Initialize workflow engine with LangGraph
        workflow_engine = WorkflowEngine(agents)
        logger.info("✅ Workflow engine initialized with LangGraph")
        
        # Initialize WebSocket manager
        websocket_manager = WebSocketManager()
        logger.info("✅ WebSocket manager initialized")
        
        # Set global instances in dependencies module
        dependencies.set_workflow_engine(workflow_engine)
        dependencies.set_websocket_manager(websocket_manager)
        dependencies.set_agents(agents)
        
        logger.info("🎯 Multi-Agent AI Content Generator is ready!")
        
        yield  # Application runs here
        
    except Exception as e:
        logger.error("❌ Application startup failed", error=str(e))
        raise
    
    finally:
        # Cleanup during shutdown
        logger.info("🛑 Shutting down Multi-Agent AI Content Generator")
        logger.info("✅ Shutdown completed successfully")

async def initialize_agents() -> Dict[str, Any]:
    """
    Initialize all specialized agents with proper dependencies and configuration.
    """
    agents = {}
    
    try:
        # Initialize Persona Research Agent
        from app.agents.persona_research_agent import PersonaResearchAgent
        agents["persona_researcher"] = PersonaResearchAgent(llm_service)
        
        # Initialize Content Strategy Agent
        from app.agents.content_strategy_agent import ContentStrategyAgent
        agents["content_strategist"] = ContentStrategyAgent(llm_service)
        
        # Initialize Creative Generation Agent
        from app.agents.creative_generation_agent import CreativeGenerationAgent
        agents["creative_generator"] = CreativeGenerationAgent(llm_service)
        
        # Initialize Quality Assurance Agent
        from app.agents.quality_assurance_agent import QualityAssuranceAgent
        agents["quality_assurance"] = QualityAssuranceAgent(llm_service)
        
        # Initialize Orchestrator Agent
        from app.agents.orchestrator_agent import OrchestratorAgent
        agents["orchestrator"] = OrchestratorAgent(llm_service, agents)
        
        # Verify all agents are healthy
        for agent_id, agent in agents.items():
            agent_info = agent.get_agent_info()
            logger.info(f"✅ Agent initialized: {agent_info['name']}", 
                       capabilities=len(agent_info['capabilities']))
        
        return agents
        
    except Exception as e:
        logger.error("Failed to initialize agents", error=str(e))
        raise

# Create FastAPI application with lifespan management
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Advanced multi-agent system for generating persona-tailored marketing content using LangGraph orchestration and specialized AI agents.",
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None,
    lifespan=lifespan
)

# Add comprehensive middleware stack
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Custom middleware for request logging and performance monitoring
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests with timing and error tracking."""
    start_time = asyncio.get_event_loop().time()
    
    # Log request start
    logger.info(
        "HTTP Request",
        method=request.method,
        url=str(request.url),
        client_ip=request.client.host if request.client else "unknown"
    )
    
    try:
        response = await call_next(request)
        
        # Calculate request duration
        duration = asyncio.get_event_loop().time() - start_time
        
        # Log successful response
        logger.info(
            "HTTP Response",
            method=request.method,
            url=str(request.url),
            status_code=response.status_code,
            duration_ms=round(duration * 1000, 2)
        )
        
        return response
        
    except Exception as e:
        # Log request error
        duration = asyncio.get_event_loop().time() - start_time
        logger.error(
            "HTTP Request Error",
            method=request.method,
            url=str(request.url),
            error=str(e),
            duration_ms=round(duration * 1000, 2)
        )
        
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "error_id": str(uuid.uuid4())}
        )

# Import routes AFTER app creation to avoid circular imports
from app.api.routes import content, agents, workflow

# Include API route modules
app.include_router(
    content.router,
    prefix=settings.API_V1_PREFIX,
    tags=["Content Generation"]
)

app.include_router(
    agents.router,
    prefix=settings.API_V1_PREFIX,
    tags=["Agent Management"]
)

app.include_router(
    workflow.router,
    prefix=settings.API_V1_PREFIX,
    tags=["Workflow Management"]
)

# Root endpoints
@app.get("/")
async def root():
    """Root endpoint with API information and health status."""
    # Get system status from dependencies
    try:
        agents = await dependencies.get_agents()
        agents_available = len(agents)
    except:
        agents_available = 0
    
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "description": "Multi-Agent AI Content Generator",
        "status": "operational",
        "agents_available": agents_available,
        "api_docs": "/docs" if settings.DEBUG else "disabled_in_production",
        "websocket_endpoint": "/ws/workflow",
        "health_check": "/health"
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check for all system components."""
    try:
        from datetime import datetime
        
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.APP_VERSION
        }
        
        # Check LLM service health
        llm_health = await llm_service.health_check()
        health_status["llm_providers"] = llm_health
        
        # Check system components through dependencies
        try:
            system_health = await dependencies.verify_system_health()
            health_status.update(system_health)
        except Exception as e:
            health_status["system_status"] = f"degraded: {str(e)}"
        
        return health_status
        
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )

@app.get("/metrics")
async def metrics():
    """System metrics and performance statistics."""
    if not settings.ENABLE_METRICS:
        raise HTTPException(status_code=404, detail="Metrics disabled")
    
    try:
        metrics_data = {
            "system": {
                "uptime": "calculated_uptime",
                "memory_usage": "system_memory",
                "cpu_usage": "system_cpu"
            }
        }
        
        # LLM service metrics
        metrics_data["llm_service"] = llm_service.get_service_stats()
        
        # Agent performance metrics
        try:
            agents = await dependencies.get_agents()
            agent_metrics = {}
            for agent_id, agent in agents.items():
                agent_metrics[agent_id] = agent.metrics.dict()
            metrics_data["agents"] = agent_metrics
        except Exception as e:
            metrics_data["agents_error"] = str(e)
        
        return metrics_data
        
    except Exception as e:
        logger.error("Failed to generate metrics", error=str(e))
        raise HTTPException(status_code=500, detail="Metrics generation failed")

# WebSocket endpoint for real-time workflow monitoring
@app.websocket("/ws/workflow")
async def websocket_workflow_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time workflow monitoring and agent communication.
    Provides live updates on workflow progress, agent status, and results.
    """
    try:
        manager = await dependencies.get_websocket_manager()
    except:
        await websocket.close(code=1011, reason="WebSocket service not available")
        return
    
    try:
        await manager.connect(websocket)
        logger.info("WebSocket client connected", client_id=websocket.client.host)
        
        # Send initial connection confirmation
        await manager.send_personal_message(websocket, {
            "type": "connection_established",
            "message": "Connected to workflow monitoring",
            "timestamp": datetime.utcnow().isoformat()
        })
        
        # Handle incoming WebSocket messages
        while True:
            try:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                # Handle different message types
                if message.get("type") == "subscribe_workflow":
                    workflow_id = message.get("workflow_id")
                    await manager.subscribe_to_workflow(websocket, workflow_id)
                
                elif message.get("type") == "get_agent_status":
                    try:
                        agents = await dependencies.get_agents()
                        agent_status = {}
                        for agent_id, agent in agents.items():
                            agent_status[agent_id] = agent.get_agent_info()
                        
                        await manager.send_personal_message(websocket, {
                            "type": "agent_status_update",
                            "data": agent_status,
                            "timestamp": datetime.utcnow().isoformat()
                        })
                    except Exception as e:
                        await manager.send_personal_message(websocket, {
                            "type": "error",
                            "message": f"Failed to get agent status: {str(e)}",
                            "timestamp": datetime.utcnow().isoformat()
                        })
                
                elif message.get("type") == "ping":
                    await manager.send_personal_message(websocket, {
                        "type": "pong",
                        "timestamp": datetime.utcnow().isoformat()
                    })
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error("WebSocket message handling error", error=str(e))
                await manager.send_personal_message(websocket, {
                    "type": "error",
                    "message": "Message handling failed",
                    "timestamp": datetime.utcnow().isoformat()
                })
    
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        logger.info("WebSocket client disconnected")
    
    except Exception as e:
        logger.error("WebSocket connection error", error=str(e))
        await websocket.close(code=1011, reason="Internal server error")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """Custom 404 handler with helpful information."""
    return JSONResponse(
        status_code=404,
        content={
            "detail": "Endpoint not found",
            "available_endpoints": {
                "content_generation": f"{settings.API_V1_PREFIX}/content/generate",
                "agent_status": f"{settings.API_V1_PREFIX}/agents/status",
                "workflow_monitoring": f"{settings.API_V1_PREFIX}/workflow/status",
                "health_check": "/health",
                "api_docs": "/docs" if settings.DEBUG else "disabled"
            }
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    """Custom 500 handler with error tracking."""
    import uuid
    error_id = str(uuid.uuid4())
    logger.error("Internal server error", error_id=error_id, error=str(exc))
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "error_id": error_id,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

# Development server runner
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=settings.DEBUG
    )
