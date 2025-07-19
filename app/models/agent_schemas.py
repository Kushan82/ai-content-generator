from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from datetime import datetime
import uuid

class AgentStatus(str, Enum):
    """
    Enumeration of possible agent execution states.
    Used to track agent lifecycle and current operational status.
    """
    INITIALIZING = "initializing"    # Agent is starting up
    IDLE = "idle"                   # Ready to accept tasks
    THINKING = "thinking"           # Processing/analyzing input
    WORKING = "working"            # Actively executing task
    COMMUNICATING = "communicating" # Exchanging data with other agents
    COMPLETED = "completed"        # Task finished successfully
    ERROR = "error"               # Encountered an error
    TIMEOUT = "timeout"           # Task exceeded time limit
    PAUSED = "paused"            # Temporarily suspended

class AgentCapability(BaseModel):
    """
    Represents a specific capability or skill that an agent possesses.
    Used for agent discovery and task routing in the multi-agent system.
    """
    name: str = Field(..., description="Capability identifier")
    description: str = Field(..., description="Human-readable capability description")
    proficiency_level: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Agent's skill level for this capability (0.0-1.0)"
    )
    required_resources: List[str] = Field(
        default_factory=list,
        description="Resources needed to execute this capability"
    )
    average_execution_time: float = Field(
        default=0.0,
        ge=0.0,
        description="Average time in seconds to execute this capability"
    )

class AgentMessage(BaseModel):
    """
    Standardized message format for inter-agent communication.
    Enables structured communication between agents in the workflow.
    """
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str = Field(..., description="ID of the sending agent")
    recipient_id: str = Field(..., description="ID of the receiving agent")
    message_type: str = Field(..., description="Type of message (task_request, response, etc.)")
    
    # Message content
    content: Dict[str, Any] = Field(..., description="Message payload")
    priority: int = Field(default=1, ge=1, le=10, description="Message priority (1=highest, 10=lowest)")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = Field(None, description="ID linking related messages")
    requires_response: bool = Field(default=False, description="Whether message requires a response")
    expiry_time: Optional[datetime] = Field(None, description="When message expires")
    
    # Performance tracking
    processing_deadline: Optional[datetime] = Field(None, description="Expected processing deadline")
    
    @validator('priority')
    def validate_priority(cls, v):
        """Ensure priority is within valid range"""
        if not 1 <= v <= 10:
            raise ValueError('Priority must be between 1 (highest) and 10 (lowest)')
        return v

class AgentMetrics(BaseModel):
    """
    Performance and operational metrics for individual agents.
    Used for monitoring, optimization, and health checking.
    """
    # Task Execution Metrics
    total_tasks_completed: int = Field(default=0, description="Total number of tasks completed")
    total_tasks_failed: int = Field(default=0, description="Total number of failed tasks")
    success_rate: float = Field(default=1.0, ge=0.0, le=1.0, description="Task success rate")
    
    # Performance Metrics
    average_response_time: float = Field(default=0.0, ge=0.0, description="Average task completion time")
    min_response_time: float = Field(default=0.0, ge=0.0, description="Fastest task completion time")
    max_response_time: float = Field(default=0.0, ge=0.0, description="Slowest task completion time")
    
    # Quality Metrics
    average_quality_score: float = Field(default=0.0, ge=0.0, le=1.0, description="Average output quality")
    total_tokens_processed: int = Field(default=0, description="Total tokens processed by agent")
    total_api_calls: int = Field(default=0, description="Total LLM API calls made")
    
    # Resource Utilization
    memory_usage_mb: float = Field(default=0.0, description="Current memory usage in MB")
    cpu_usage_percent: float = Field(default=0.0, ge=0.0, le=100.0, description="CPU usage percentage")
    
    # Temporal Metrics
    uptime_seconds: float = Field(default=0.0, description="Agent uptime in seconds")
    last_activity: Optional[datetime] = Field(None, description="Timestamp of last activity")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Error Tracking
    recent_errors: List[str] = Field(default_factory=list, description="Recent error messages")
    error_categories: Dict[str, int] = Field(default_factory=dict, description="Error count by category")
    
    def update_success_rate(self):
        """Recalculate success rate based on completed and failed tasks"""
        total_tasks = self.total_tasks_completed + self.total_tasks_failed
        if total_tasks > 0:
            self.success_rate = self.total_tasks_completed / total_tasks
        else:
            self.success_rate = 1.0

class AgentResponse(BaseModel):
    """
    Standardized response format from individual agents.
    Provides consistent structure for agent outputs and metadata.
    """
    # Agent Identification
    agent_id: str = Field(..., description="Unique agent identifier")
    agent_name: str = Field(..., description="Human-readable agent name")
    agent_version: str = Field(default="1.0.0", description="Agent implementation version")
    
    # Task Information  
    task_id: str = Field(..., description="Unique task identifier")
    task_type: str = Field(..., description="Type of task performed")
    
    # Response Content
    content: Dict[str, Any] = Field(..., description="Agent's response data")
    
    # Quality and Confidence Metrics
    confidence_score: float = Field(
        ge=0.0, 
        le=1.0, 
        description="Agent's confidence in the response quality"
    )
    quality_indicators: Dict[str, Any] = Field(
        default_factory=dict,
        description="Specific quality metrics for the response"
    )
    
    # Performance Metrics
    processing_time: float = Field(ge=0.0, description="Time taken to process the task")
    tokens_used: int = Field(default=0, description="Number of tokens consumed")
    api_calls_made: int = Field(default=0, description="Number of API calls made")
    
    # Status and Error Handling
    status: AgentStatus = Field(default=AgentStatus.COMPLETED)
    error_message: Optional[str] = Field(None, description="Error message if task failed")
    warnings: List[str] = Field(default_factory=list, description="Non-critical warnings")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional response metadata"
    )
    
    # Dependencies and References
    depends_on: List[str] = Field(
        default_factory=list,
        description="Other agent responses this depends on"
    )
    references: List[str] = Field(
        default_factory=list,
        description="External references or sources used"
    )
    
    @validator('confidence_score')
    def validate_confidence(cls, v):
        """Ensure confidence score is reasonable"""
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence score must be between 0.0 and 1.0')
        return v
    
    def is_successful(self) -> bool:
        """Check if the agent response indicates success"""
        return self.status == AgentStatus.COMPLETED and self.error_message is None
    
    def has_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if the response has high confidence above threshold"""
        return self.confidence_score >= threshold

class AgentHealthCheck(BaseModel):
    """
    Health check response for monitoring agent status and capabilities.
    Used by the monitoring system to track agent availability and performance.
    """
    agent_id: str = Field(..., description="Agent identifier")
    agent_name: str = Field(..., description="Agent name")
    
    # Health Status
    is_healthy: bool = Field(..., description="Overall health status")
    health_score: float = Field(ge=0.0, le=1.0, description="Health score (0-1)")
    status: AgentStatus = Field(..., description="Current agent status")
    
    # Capabilities
    available_capabilities: List[AgentCapability] = Field(
        default_factory=list,
        description="Currently available capabilities"
    )
    
    # Performance Indicators
    current_load: float = Field(ge=0.0, le=1.0, description="Current workload (0-1)")
    response_time_ms: float = Field(ge=0.0, description="Health check response time")
    
    # Resource Status
    memory_usage_mb: float = Field(ge=0.0, description="Memory usage")
    cpu_usage_percent: float = Field(ge=0.0, le=100.0, description="CPU usage")
    
    # Timestamps
    last_successful_task: Optional[datetime] = Field(None, description="Last successful task completion")
    last_error: Optional[datetime] = Field(None, description="Last error occurrence")
    uptime_seconds: float = Field(ge=0.0, description="Agent uptime")
    
    # Issues and Warnings
    active_issues: List[str] = Field(default_factory=list, description="Current issues")
    warnings: List[str] = Field(default_factory=list, description="Current warnings")
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
