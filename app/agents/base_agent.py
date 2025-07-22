"""Enhanced base agent with better error handling and monitoring"""
import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from contextlib import asynccontextmanager

from pydantic import BaseModel, Field

from app.models.agent_schemas import (
    AgentStatus,
    AgentResponse,
    AgentMetrics,
    AgentCapability,
    AgentMessage
)
from app.services.llm_service import LLMService, LLMRequest
from app.core.logging import get_agent_logger, AgentLogger
from app.core.config import settings
from app.core.exceptions import AgentError, MultiAgentError
from app.core.monitoring import AgentMetricsCollector
from app.core.types import AgentID, TaskID, WorkflowID

class TaskRequest(BaseModel):
    """Enhanced task request with validation"""
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = Field(..., description="Type of task to perform")
    input_data: Dict[str, Any] = Field(..., description="Input data for the task")
    
    # Task configuration
    priority: int = Field(default=5, ge=1, le=10, description="Task priority (1=highest)")
    timeout: int = Field(default=settings.AGENT_TIMEOUT, description="Task timeout in seconds")
    retry_attempts: int = Field(default=settings.MAX_RETRY_ATTEMPTS, description="Max retry attempts")
    
    # Context and metadata
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    workflow_id: Optional[str] = Field(None, description="Associated workflow ID")
    parent_task_id: Optional[str] = Field(None, description="Parent task if this is a subtask")
    
    # Quality requirements
    min_confidence: float = Field(default=0.7, ge=0.0, le=1.0, description="Minimum confidence threshold")
    quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Minimum quality threshold")

class BaseAgent(ABC):
    """
    Enhanced abstract base class for all agents in the multi-agent system.
    Combines original functionality with better error handling and monitoring.
    """
    
    def __init__(
        self,
        agent_id: str,
        name: str,
        role: str,
        llm_service: LLMService,
        capabilities: Optional[List[AgentCapability]] = None
    ):
        # Agent identification
        self.agent_id = agent_id
        self.name = name
        self.role = role
        self.version = "2.0.0"  # Updated version for enhanced agent
        
        # Core services
        self.llm_service = llm_service
        self.logger = get_agent_logger(agent_id, name)
        
        # Enhanced monitoring
        self.metrics_collector = AgentMetricsCollector(agent_id)
        self.metrics = AgentMetrics()  # Keep original metrics for compatibility
        
        # Agent state
        self.status = AgentStatus.INITIALIZING
        self.capabilities = capabilities or []
        self.current_task: Optional[TaskRequest] = None
        
        # Performance tracking
        self.start_time = time.time()
        
        # Communication (original functionality)
        self.message_handlers: Dict[str, Callable] = {}
        self.subscribers: List[str] = []
        
        # Memory and context (original functionality)
        self.memory: Dict[str, Any] = {}
        self.context_window: List[Dict[str, Any]] = []
        self.max_context_size = 50
        
        # Enhanced error handling
        self.error_recovery_strategies: Dict[str, Callable] = {}
        self.last_error: Optional[Exception] = None
        
        self.logger.info(f"Enhanced agent {name} initialized", role=role, capabilities=len(self.capabilities))
        self._initialize_agent()
    
    def _initialize_agent(self):
        """Initialize agent-specific resources and configurations"""
        self.status = AgentStatus.IDLE
        self.metrics.created_at = datetime.utcnow()
        
        # Register default message handlers
        self.message_handlers.update({
            "task_request": self._handle_task_request,
            "status_check": self._handle_status_check,
            "health_check": self._handle_health_check
        })
        
        self.logger.info("Enhanced agent initialization completed", agent_id=self.agent_id)
    
    @asynccontextmanager
    async def execution_context(self, task: TaskRequest):
        """Context manager for task execution with proper cleanup and monitoring"""
        self.status = AgentStatus.WORKING
        self.current_task = task
        start_time = time.time()
        
        try:
            # Start monitoring
            self.metrics_collector.start_task(task.task_id)
            
            self.logger.task_start(
                task_id=task.task_id,
                task_type=task.task_type,
                priority=task.priority,
                workflow_id=task.workflow_id
            )
            
            yield
            
        except Exception as e:
            # Record error in enhanced monitoring
            self.metrics_collector.record_error(task.task_id, str(e))
            self.status = AgentStatus.ERROR
            self.last_error = e
            
            # Update original metrics for compatibility
            self._update_failure_metrics()
            
            # Log error
            self.logger.task_error(
                task_id=task.task_id,
                error=e,
                processing_time=time.time() - start_time
            )
            
            # Raise enhanced error
            raise AgentError(
                f"Task execution failed: {str(e)}", 
                error_code="TASK_EXECUTION_FAILED",
                details={"task_id": task.task_id, "agent_id": self.agent_id}
            )
            
        finally:
            execution_time = time.time() - start_time
            
            # Complete monitoring
            if task.task_id in self.metrics_collector.task_metrics:
                confidence_score = self._calculate_confidence_from_context(task, execution_time)
                self.metrics_collector.complete_task(task.task_id, execution_time, confidence_score)
            
            # Update original metrics
            if self.status != AgentStatus.ERROR:
                self._update_success_metrics(execution_time)
            
            # Reset state
            self.current_task = None
            if self.status not in [AgentStatus.ERROR, AgentStatus.TIMEOUT]:
                self.status = AgentStatus.IDLE
    
    async def execute_task(self, task: TaskRequest) -> AgentResponse:
        """
        Enhanced task execution combining original functionality with new monitoring
        """
        # Pre-execution validation (enhanced)
        self._validate_task_input(task)
        
        # Check capability (original functionality)
        if not self._can_handle_task(task):
            raise AgentError(
                f"Agent {self.name} cannot handle task type: {task.task_type}",
                error_code="CAPABILITY_MISMATCH",
                details={"agent_id": self.agent_id, "task_type": task.task_type}
            )
        
        try:
            async with self.execution_context(task):
                # Execute with timeout (original functionality enhanced)
                result = await asyncio.wait_for(
                    self.process_task(task),
                    timeout=task.timeout
                )
                
                # Output validation (enhanced)
                self._validate_task_output(result, task)
                
                # Create response (original functionality)
                processing_time = time.time() - self.metrics_collector.task_metrics.get(
                    task.task_id, type('obj', (object,), {'start_time': datetime.utcnow()})
                ).start_time.timestamp() if task.task_id in self.metrics_collector.task_metrics else 0
                
                response = self._create_response(task, result, processing_time, AgentStatus.COMPLETED)
                
                # Quality validation (original functionality)
                if not self._validate_response_quality(response, task):
                    self.logger.warning(
                        "Response quality below threshold",
                        task_id=task.task_id,
                        confidence=response.confidence_score,
                        threshold=task.min_confidence
                    )
                
                self.logger.task_complete(
                    task_id=task.task_id,
                    duration=processing_time,
                    success=True,
                    confidence=response.confidence_score
                )
                
                return response
                
        except asyncio.TimeoutError:
            self.status = AgentStatus.TIMEOUT
            error_msg = f"Task timed out after {task.timeout} seconds"
            
            raise AgentError(
                error_msg,
                error_code="TASK_TIMEOUT",
                details={"task_id": task.task_id, "timeout": task.timeout}
            )
            
        except AgentError:
            # Re-raise AgentError as-is
            raise
            
        except Exception as e:
            # Convert other exceptions to AgentError
            raise AgentError(
                f"Unexpected error during task execution: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={"task_id": task.task_id, "original_error": str(e)}
            )
    
    def _validate_task_input(self, task: TaskRequest):
        """Enhanced input validation"""
        if not task.input_data:
            raise AgentError(
                "Task input data is required", 
                error_code="MISSING_INPUT",
                details={"task_id": task.task_id}
            )
        
        # Additional validation can be added here
        if task.priority < 1 or task.priority > 10:
            raise AgentError(
                "Task priority must be between 1 and 10",
                error_code="INVALID_PRIORITY",
                details={"task_id": task.task_id, "priority": task.priority}
            )
    
    def _validate_task_output(self, result: Dict[str, Any], task: TaskRequest):
        """Enhanced output validation"""
        if not result:
            raise AgentError(
                "Task produced no output", 
                error_code="EMPTY_OUTPUT",
                details={"task_id": task.task_id}
            )
        
        # Check for required output fields
        if "content" not in result:
            raise AgentError(
                "Task output missing required 'content' field",
                error_code="INVALID_OUTPUT_FORMAT",
                details={"task_id": task.task_id, "result_keys": list(result.keys())}
            )
    
    # Keep all original functionality methods
    def _can_handle_task(self, task: TaskRequest) -> bool:
        """Check if the agent has the capabilities to handle the task (original)"""
        required_capability = task.task_type
        
        for capability in self.capabilities:
            if capability.name == required_capability:
                return capability.proficiency_level >= 0.5
        
        return len(self.capabilities) == 0 or hasattr(self, f"_handle_{task.task_type}")
    
    def _create_response(
        self,
        task: TaskRequest,
        result: Dict[str, Any],
        processing_time: float,
        status: AgentStatus
    ) -> AgentResponse:
        """Create a standardized agent response (original with enhancements)"""
        confidence = self._calculate_confidence(task, result, processing_time)
        
        return AgentResponse(
            agent_id=self.agent_id,
            agent_name=self.name,
            agent_version=self.version,
            task_id=task.task_id,
            task_type=task.task_type,
            content=result,
            confidence_score=confidence,
            processing_time=processing_time,
            status=status,
            tokens_used=result.get("tokens_used", 0),
            api_calls_made=result.get("api_calls", 0),
            metadata={
                "workflow_id": task.workflow_id,
                "agent_role": self.role,
                "task_priority": task.priority,
                "retry_count": task.retry_attempts,
                "enhanced_agent": True  # Flag for enhanced agent
            }
        )
    
    def _calculate_confidence_from_context(self, task: TaskRequest, execution_time: float) -> float:
        """Calculate confidence for metrics collector"""
        # Use original confidence calculation
        base_confidence = 0.8
        time_factor = min(execution_time / 10.0, 1.0)
        complexity_factor = 0.9 if task.task_type in ["research", "strategy"] else 1.0
        
        confidence = base_confidence * time_factor * complexity_factor
        return min(max(confidence, 0.1), 1.0)
    
    def get_enhanced_metrics(self) -> Dict[str, Any]:
        """Get enhanced metrics from the new collector"""
        return self.metrics_collector.get_performance_summary()
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Enhanced agent information including new metrics"""
        original_info = {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role,
            "version": self.version,
            "status": self.status.value,
            "capabilities": [cap.dict() for cap in self.capabilities],
            "metrics": self.metrics.dict(),
            "uptime": time.time() - self.start_time,
            "memory_size": len(self.memory),
            "context_window_size": len(self.context_window)
        }
        
        # Add enhanced metrics
        original_info["enhanced_metrics"] = self.get_enhanced_metrics()
        original_info["is_enhanced"] = True
        
        return original_info
    
    # Keep all original methods for backward compatibility
    def _calculate_confidence(self, task: TaskRequest, result: Dict[str, Any], processing_time: float) -> float:
        """Original confidence calculation"""
        base_confidence = 0.8
        time_factor = min(processing_time / 10.0, 1.0)
        complexity_factor = 0.9 if task.task_type in ["research", "strategy"] else 1.0
        content_factor = min(len(str(result.get("content", ""))) / 1000.0, 1.0)
        
        confidence = base_confidence * time_factor * complexity_factor * content_factor
        return min(max(confidence, 0.1), 1.0)
    
    def _validate_response_quality(self, response: AgentResponse, task: TaskRequest) -> bool:
        """Original quality validation"""
        return (
            response.confidence_score >= task.min_confidence and
            response.status == AgentStatus.COMPLETED and
            response.error_message is None
        )
    
    def _update_success_metrics(self, processing_time: float):
        """Original success metrics update"""
        self.metrics.total_tasks_completed += 1
        self.metrics.last_activity = datetime.utcnow()
        
        total_time = (self.metrics.average_response_time * (self.metrics.total_tasks_completed - 1) + processing_time)
        self.metrics.average_response_time = total_time / self.metrics.total_tasks_completed
        
        if self.metrics.min_response_time == 0 or processing_time < self.metrics.min_response_time:
            self.metrics.min_response_time = processing_time
            
        if processing_time > self.metrics.max_response_time:
            self.metrics.max_response_time = processing_time
        
        self.metrics.update_success_rate()
    
    def _update_failure_metrics(self):
        """Original failure metrics update"""
        self.metrics.total_tasks_failed += 1
        self.metrics.update_success_rate()
    
    # All other original methods remain unchanged...
    async def _attempt_error_recovery(self, task: TaskRequest, error: Exception) -> Optional[AgentResponse]:
        """Original error recovery (kept for compatibility)"""
        error_type = type(error).__name__
        
        if error_type in self.error_recovery_strategies:
            try:
                self.logger.info(f"Attempting error recovery for {error_type}", task_id=task.task_id)
                recovery_strategy = self.error_recovery_strategies[error_type]
                return await recovery_strategy(task, error)
            except Exception as recovery_error:
                self.logger.error(
                    "Error recovery failed",
                    task_id=task.task_id,
                    recovery_error=str(recovery_error)
                )
        
        return None
    async def _handle_task_request(self, message):
        """Handle task request message"""
        return {"status": "task_received"}

    async def _handle_status_check(self, message):
        """Handle status check message"""
        return {"status": self.status.value}

    async def _handle_health_check(self, message):
        """Handle health check message"""
        return {"healthy": True}

    # Keep all other original methods for full backward compatibility...
    async def send_message(self, recipient_id: str, message: AgentMessage) -> bool:
        """Send a message to another agent (original)"""
        self.logger.info(
            "Sending message to agent",
            recipient=recipient_id,
            message_type=message.message_type,
            correlation_id=message.correlation_id
        )
        return True
    
    def add_memory(self, key: str, value: Any, ttl: Optional[int] = None):
        """Add information to agent's memory (original)"""
        self.memory[key] = {
            "value": value,
            "timestamp": datetime.utcnow(),
            "ttl": ttl
        }
        
        self.context_window.append({
            "type": "memory_update",
            "key": key,
            "timestamp": datetime.utcnow()
        })
        
        if len(self.context_window) > self.max_context_size:
            self.context_window.pop(0)
    
    def get_memory(self, key: str) -> Optional[Any]:
        """Retrieve information from agent's memory (original)"""
        if key in self.memory:
            memory_item = self.memory[key]
            
            if memory_item.get("ttl"):
                age = (datetime.utcnow() - memory_item["timestamp"]).seconds
                if age > memory_item["ttl"]:
                    del self.memory[key]
                    return None
            
            return memory_item["value"]
        
        return None
    
    @abstractmethod
    async def process_task(self, task: TaskRequest) -> Dict[str, Any]:
        """
        Core task processing method that must be implemented by each agent.
        This is where the agent's specific intelligence and capabilities are executed.
        """
        pass


# Updated factory function for enhanced agents
def create_agent(
    agent_class,
    agent_id: str,
    name: str,
    role: str,
    capabilities: List[AgentCapability],
    llm_service: LLMService
):
    """Factory function to create enhanced agents with standard configuration"""
    return agent_class(
        agent_id=agent_id,
        name=name,
        role=role,
        llm_service=llm_service,
        capabilities=capabilities
    )
