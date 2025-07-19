"""
Data models and schemas for the Multi-Agent AI Content Generator.
Provides type-safe data structures for agent communication, workflow management,
and content generation across the entire application.
"""

from .agent_schemas import (
    AgentStatus,
    AgentMessage, 
    AgentResponse,
    AgentCapability,
    AgentMetrics
)

from .workflow_schemas import (
    WorkflowState,
    WorkflowStatus,
    WorkflowStep,
    WorkflowResult,
    AgentCommunication
)

from .content_schemas import (
    PersonaType,
    ContentType,
    Persona,
    ContentRequest,
    ContentResponse,
    QualityMetrics
)

__all__ = [
    # Agent schemas
    "AgentStatus",
    "AgentMessage", 
    "AgentResponse",
    "AgentCapability",
    "AgentMetrics",
    
    # Workflow schemas
    "WorkflowState",
    "WorkflowStatus", 
    "WorkflowStep",
    "WorkflowResult",
    "AgentCommunication",
    
    # Content schemas
    "PersonaType",
    "ContentType",
    "Persona",
    "ContentRequest", 
    "ContentResponse",
    "QualityMetrics"
]
