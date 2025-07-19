from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Union
from enum import Enum
from datetime import datetime
import uuid
from app.models.agent_schemas import AgentResponse, AgentStatus

class WorkflowStatus(str, Enum):
    """
    Enumeration of possible workflow execution states.
    Tracks the overall status of multi-agent workflow execution.
    """
    PENDING = "pending"           # Workflow queued but not started
    INITIALIZING = "initializing" # Setting up agents and resources
    RUNNING = "running"          # Active execution in progress
    WAITING = "waiting"          # Waiting for agent responses
    PAUSED = "paused"           # Temporarily suspended
    COMPLETED = "completed"      # Successfully finished
    FAILED = "failed"           # Failed with errors
    CANCELLED = "cancelled"      # Manually cancelled
    TIMEOUT = "timeout"         # Exceeded maximum execution time

class WorkflowStep(BaseModel):
    """
    Represents a single step in the multi-agent workflow.
    Each step corresponds to an agent task or coordination point.
    """
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    step_name: str = Field(..., description="Human-readable step name")
    step_type: str = Field(..., description="Type of step (agent_task, coordination, etc.)")
    
    # Agent Assignment
    assigned_agent_id: str = Field(..., description="Agent responsible for this step")
    agent_name: str = Field(..., description="Name of assigned agent")
    
    # Dependencies
    depends_on: List[str] = Field(
        default_factory=list,
        description="Step IDs this step depends on"
    )
    blocks: List[str] = Field(
        default_factory=list, 
        description="Step IDs that depend on this step"
    )
    
    # Execution Details
    status: AgentStatus = Field(default=AgentStatus.IDLE)
    start_time: Optional[datetime] = Field(None, description="Step start timestamp")
    end_time: Optional[datetime] = Field(None, description="Step completion timestamp")
    duration_seconds: Optional[float] = Field(None, description="Step execution duration")
    
    # Input and Output
    input_data: Dict[str, Any] = Field(default_factory=dict, description="Input data for the step")
    output_data: Dict[str, Any] = Field(default_factory=dict, description="Output data from the step")
    
    # Quality and Performance
    confidence_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    retry_count: int = Field(default=0, description="Number of retry attempts")
    
    # Error Handling
    error_message: Optional[str] = Field(None, description="Error message if step failed")
    warnings: List[str] = Field(default_factory=list, description="Step warnings")
    
    def calculate_duration(self) -> Optional[float]:
        """Calculate step duration if start and end times are available"""
        if self.start_time and self.end_time:
            self.duration_seconds = (self.end_time - self.start_time).total_seconds()
            return self.duration_seconds
        return None
    
    def is_ready_to_execute(self) -> bool:
        """Check if all dependencies are met and step can execute"""
        return self.status == AgentStatus.IDLE and len(self.depends_on) == 0

class AgentCommunication(BaseModel):
    """
    Tracks communication between agents during workflow execution.
    Provides audit trail and debugging information for agent interactions.
    """
    communication_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = Field(..., description="Associated workflow ID")
    
    # Communication Details
    sender_agent_id: str = Field(..., description="Sending agent ID")
    receiver_agent_id: str = Field(..., description="Receiving agent ID")
    message_type: str = Field(..., description="Type of communication")
    
    # Content
    message_content: Dict[str, Any] = Field(..., description="Communication payload")
    response_content: Optional[Dict[str, Any]] = Field(None, description="Response payload")
    
    # Timing
    sent_at: datetime = Field(default_factory=datetime.utcnow)
    received_at: Optional[datetime] = Field(None, description="When message was received")
    responded_at: Optional[datetime] = Field(None, description="When response was sent")
    
    # Status
    delivery_status: str = Field(default="sent", description="Delivery status")
    requires_response: bool = Field(default=False)
    response_received: bool = Field(default=False)
    
    # Quality
    communication_success: bool = Field(default=True)
    error_message: Optional[str] = Field(None)

class WorkflowState(BaseModel):
    """
    Complete state representation of a multi-agent workflow execution.
    Used by LangGraph for state management and persistence.
    """
    # Workflow Identification
    workflow_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_name: str = Field(..., description="Human-readable workflow name")
    workflow_type: str = Field(..., description="Type of workflow being executed")
    workflow_version: str = Field(default="1.0.0", description="Workflow definition version")
    
    # Request Information
    original_request: Dict[str, Any] = Field(..., description="Original user request")
    request_id: str = Field(..., description="Associated request ID")
    
    # Execution State
    status: WorkflowStatus = Field(default=WorkflowStatus.PENDING)
    current_step: str = Field(default="initialization", description="Current execution step")
    
    # Agent Results Storage
    agent_responses: Dict[str, AgentResponse] = Field(
        default_factory=dict,
        description="Responses from each agent indexed by agent_id"
    )
    
    # Step Management
    workflow_steps: List[WorkflowStep] = Field(
        default_factory=list,
        description="All steps in the workflow"
    )
    completed_steps: List[str] = Field(
        default_factory=list,
        description="IDs of completed steps"
    )
    
    # Inter-Agent Communication
    communications: List[AgentCommunication] = Field(
        default_factory=list,
        description="Record of all agent communications"
    )
    
    # Timing and Performance
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(None, description="Workflow start time")
    completed_at: Optional[datetime] = Field(None, description="Workflow completion time")
    total_duration_seconds: Optional[float] = Field(None, description="Total execution time")
    
    # Quality Metrics
    overall_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    overall_confidence_score: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Error Handling and Retries
    errors: List[str] = Field(default_factory=list, description="Workflow errors")
    warnings: List[str] = Field(default_factory=list, description="Workflow warnings")
    retry_count: int = Field(default=0, description="Number of retry attempts")
    max_retries: int = Field(default=3, description="Maximum allowed retries")
    
    # Resource Tracking
    total_tokens_used: int = Field(default=0, description="Total tokens consumed")
    total_api_calls: int = Field(default=0, description="Total API calls made")
    estimated_cost: float = Field(default=0.0, description="Estimated execution cost")
    
    # Metadata and Context
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional workflow metadata"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Workflow execution context"
    )
    
    def add_agent_response(self, agent_id: str, response: AgentResponse):
        """Add an agent response to the workflow state"""
        self.agent_responses[agent_id] = response
        
        # Update aggregated metrics
        self.total_tokens_used += response.tokens_used
        self.total_api_calls += response.api_calls_made
        
        # Update quality scores (weighted average)
        if len(self.agent_responses) > 0:
            total_confidence = sum(r.confidence_score for r in self.agent_responses.values())
            self.overall_confidence_score = total_confidence / len(self.agent_responses)
    
    def calculate_total_duration(self) -> Optional[float]:
        """Calculate total workflow duration"""
        if self.started_at and self.completed_at:
            self.total_duration_seconds = (self.completed_at - self.started_at).total_seconds()
            return self.total_duration_seconds
        return None
    
    def get_next_ready_steps(self) -> List[WorkflowStep]:
        """Get workflow steps that are ready to execute"""
        ready_steps = []
        for step in self.workflow_steps:
            if step.is_ready_to_execute():
                # Check if all dependencies are completed
                deps_completed = all(dep_id in self.completed_steps for dep_id in step.depends_on)
                if deps_completed:
                    ready_steps.append(step)
        return ready_steps
    
    def is_workflow_complete(self) -> bool:
        """Check if all workflow steps are completed"""
        return len(self.completed_steps) == len(self.workflow_steps)
    
    def get_agent_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summary for each agent involved"""
        summary = {}
        for agent_id, response in self.agent_responses.items():
            summary[agent_id] = {
                "processing_time": response.processing_time,
                "confidence_score": response.confidence_score,
                "tokens_used": response.tokens_used,
                "api_calls": response.api_calls_made,
                "success": response.is_successful()
            }
        return summary

class WorkflowResult(BaseModel):
    """
    Final result of a completed multi-agent workflow execution.
    Contains all outputs, metrics, and execution details.
    """
    # Result Identification
    result_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    workflow_id: str = Field(..., description="Associated workflow ID")
    request_id: str = Field(..., description="Original request ID")
    
    # Final Outputs
    final_result: Dict[str, Any] = Field(..., description="Primary workflow output")
    intermediate_results: Dict[str, Any] = Field(
        default_factory=dict,
        description="Intermediate results from workflow steps"
    )
    
    # Execution Summary
    workflow_state: WorkflowState = Field(..., description="Final workflow state")
    execution_successful: bool = Field(..., description="Whether workflow completed successfully")
    
    # Performance Metrics
    total_execution_time: float = Field(ge=0.0, description="Total execution time in seconds")
    agent_performance: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Performance metrics for each agent"
    )
    
    # Quality Assessment
    overall_quality_score: float = Field(ge=0.0, le=1.0, description="Overall result quality")
    quality_breakdown: Dict[str, float] = Field(
        default_factory=dict,
        description="Quality scores by component"
    )
    
    # Resource Utilization
    total_cost: float = Field(ge=0.0, description="Total execution cost")
    resource_usage: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resource utilization summary"
    )
    
    # Timestamps
    completed_at: datetime = Field(default_factory=datetime.utcnow)
    
    # Recommendations and Insights
    recommendations: List[str] = Field(
        default_factory=list,
        description="Optimization recommendations"
    )
    insights: Dict[str, Any] = Field(
        default_factory=dict,
        description="Execution insights and analysis"
    )
