import asyncio
import time
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable

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

class TaskRequest(BaseModel):
    """
    Represents a task request sent to an agent.
    Contains all necessary information for task execution.
    """
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
    Abstract base class for all agents in the multi-agent system.
    Provides common functionality for task execution, communication,
    performance tracking, and error handling.
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
        self.version = "1.0.0"
        
        # Core services
        self.llm_service = llm_service
        self.logger = get_agent_logger(agent_id, name)
        
        # Agent state
        self.status = AgentStatus.INITIALIZING
        self.capabilities = capabilities or []
        self.current_task: Optional[TaskRequest] = None
        
        # Performance tracking
        self.metrics = AgentMetrics()
        self.start_time = time.time()
        
        # Communication
        self.message_handlers: Dict[str, Callable] = {}
        self.subscribers: List[str] = []  # Other agents subscribing to this agent's outputs
        
        # Memory and context (simple in-memory storage)
        self.memory: Dict[str, Any] = {}
        self.context_window: List[Dict[str, Any]] = []
        self.max_context_size = 50
        
        # Error handling
        self.error_recovery_strategies: Dict[str, Callable] = {}
        self.last_error: Optional[Exception] = None
        
        self.logger.info(f"Agent {name} initialized", role=role, capabilities=len(self.capabilities))
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
        
        self.logger.info("Agent initialization completed", agent_id=self.agent_id)
    
    @abstractmethod
    async def process_task(self, task: TaskRequest) -> Dict[str, Any]:
        """
        Core task processing method that must be implemented by each agent.
        This is where the agent's specific intelligence and capabilities are executed.
        """
        pass
    
    async def execute_task(self, task: TaskRequest) -> AgentResponse:
        """
        Main task execution method with comprehensive error handling,
        performance tracking, and quality validation.
        """
        start_time = time.time()
        self.current_task = task
        self.status = AgentStatus.THINKING
        
        self.logger.task_start(
            task_id=task.task_id,
            task_type=task.task_type,
            priority=task.priority,
            workflow_id=task.workflow_id
        )
        
        try:
            # Pre-execution validation
            if not self._can_handle_task(task):
                raise ValueError(f"Agent {self.name} cannot handle task type: {task.task_type}")
            
            self.status = AgentStatus.WORKING
            
            # Execute the task with timeout
            result = await asyncio.wait_for(
                self.process_task(task),
                timeout=task.timeout
            )
            
            # Post-processing and validation
            processing_time = time.time() - start_time
            response = self._create_response(task, result, processing_time, AgentStatus.COMPLETED)
            
            # Quality validation
            if not self._validate_response_quality(response, task):
                self.logger.warning(
                    "Response quality below threshold",
                    task_id=task.task_id,
                    confidence=response.confidence_score,
                    threshold=task.min_confidence
                )
            
            # Update metrics
            self._update_success_metrics(processing_time)
            
            self.logger.task_complete(
                task_id=task.task_id,
                duration=processing_time,
                success=True,
                confidence=response.confidence_score
            )
            
            return response
            
        except asyncio.TimeoutError:
            processing_time = time.time() - start_time
            self.status = AgentStatus.TIMEOUT
            error_msg = f"Task timed out after {task.timeout} seconds"
            
            self.logger.task_error(
                task_id=task.task_id,
                error=Exception(error_msg),
                processing_time=processing_time
            )
            
            return self._create_error_response(task, error_msg, processing_time)
            
        except Exception as e:
            processing_time = time.time() - start_time
            self.status = AgentStatus.ERROR
            self.last_error = e
            
            # Update failure metrics
            self._update_failure_metrics()
            
            self.logger.task_error(
                task_id=task.task_id,
                error=e,
                processing_time=processing_time
            )
            
            # Attempt error recovery
            recovery_result = await self._attempt_error_recovery(task, e)
            if recovery_result:
                return recovery_result
            
            return self._create_error_response(task, str(e), processing_time)
            
        finally:
            self.current_task = None
            if self.status not in [AgentStatus.ERROR, AgentStatus.TIMEOUT]:
                self.status = AgentStatus.IDLE
    
    def _can_handle_task(self, task: TaskRequest) -> bool:
        """Check if the agent has the capabilities to handle the task"""
        required_capability = task.task_type
        
        # Check if agent has the required capability
        for capability in self.capabilities:
            if capability.name == required_capability:
                return capability.proficiency_level >= 0.5  # Minimum proficiency threshold
        
        # Some agents might handle tasks without explicit capabilities (generic agents)
        return len(self.capabilities) == 0 or hasattr(self, f"_handle_{task.task_type}")
    
    def _create_response(
        self, 
        task: TaskRequest, 
        result: Dict[str, Any], 
        processing_time: float, 
        status: AgentStatus
    ) -> AgentResponse:
        """Create a standardized agent response"""
        # Calculate confidence based on task success and processing time
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
                "retry_count": task.retry_attempts
            }
        )
    
    def _create_error_response(
        self, 
        task: TaskRequest, 
        error_message: str, 
        processing_time: float
    ) -> AgentResponse:
        """Create an error response"""
        return AgentResponse(
            agent_id=self.agent_id,
            agent_name=self.name,
            agent_version=self.version,
            task_id=task.task_id,
            task_type=task.task_type,
            content={"error": error_message},
            confidence_score=0.0,
            processing_time=processing_time,
            status=AgentStatus.ERROR,
            error_message=error_message,
            metadata={
                "workflow_id": task.workflow_id,
                "agent_role": self.role,
                "failure_type": "execution_error"
            }
        )
    
    def _calculate_confidence(
        self, 
        task: TaskRequest, 
        result: Dict[str, Any], 
        processing_time: float
    ) -> float:
        """Calculate confidence score based on various factors"""
        base_confidence = 0.8
        
        # Adjust based on processing time (faster might be less thorough)
        time_factor = min(processing_time / 10.0, 1.0)  # Normalize to 10 seconds
        
        # Adjust based on task complexity (more complex tasks might have lower confidence)
        complexity_factor = 0.9 if task.task_type in ["research", "strategy"] else 1.0
        
        # Adjust based on result content quality
        content_factor = min(len(str(result.get("content", ""))) / 1000.0, 1.0)
        
        confidence = base_confidence * time_factor * complexity_factor * content_factor
        return min(max(confidence, 0.1), 1.0)  # Clamp between 0.1 and 1.0
    
    def _validate_response_quality(self, response: AgentResponse, task: TaskRequest) -> bool:
        """Validate that the response meets quality thresholds"""
        return (
            response.confidence_score >= task.min_confidence and
            response.status == AgentStatus.COMPLETED and
            response.error_message is None
        )
    
    def _update_success_metrics(self, processing_time: float):
        """Update success-related metrics"""
        self.metrics.total_tasks_completed += 1
        self.metrics.last_activity = datetime.utcnow()
        
        # Update timing metrics
        total_time = (self.metrics.average_response_time * (self.metrics.total_tasks_completed - 1) + processing_time)
        self.metrics.average_response_time = total_time / self.metrics.total_tasks_completed
        
        if self.metrics.min_response_time == 0 or processing_time < self.metrics.min_response_time:
            self.metrics.min_response_time = processing_time
            
        if processing_time > self.metrics.max_response_time:
            self.metrics.max_response_time = processing_time
        
        self.metrics.update_success_rate()
    
    def _update_failure_metrics(self):
        """Update failure-related metrics"""
        self.metrics.total_tasks_failed += 1
        self.metrics.update_success_rate()
    
    async def _attempt_error_recovery(
        self, 
        task: TaskRequest, 
        error: Exception
    ) -> Optional[AgentResponse]:
        """Attempt to recover from errors using registered strategies"""
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
    
    async def send_message(self, recipient_id: str, message: AgentMessage) -> bool:
        """Send a message to another agent (placeholder for inter-agent communication)"""
        # In a real implementation, this would use a message bus or direct communication
        self.logger.info(
            "Sending message to agent",
            recipient=recipient_id,
            message_type=message.message_type,
            correlation_id=message.correlation_id
        )
        return True
    
    async def _handle_task_request(self, message: AgentMessage) -> Optional[AgentMessage]:
        """Handle incoming task request messages"""
        try:
            task_data = message.content
            task = TaskRequest(**task_data)
            response = await self.execute_task(task)
            
            return AgentMessage(
                sender_id=self.agent_id,
                recipient_id=message.sender_id,
                message_type="task_response",
                content=response.dict(),
                correlation_id=message.correlation_id
            )
        except Exception as e:
            self.logger.error("Failed to handle task request", error=str(e))
            return None
    
    async def _handle_status_check(self, message: AgentMessage) -> AgentMessage:
        """Handle status check requests"""
        status_info = {
            "agent_id": self.agent_id,
            "name": self.name,
            "status": self.status.value,
            "current_task": self.current_task.task_id if self.current_task else None,
            "metrics": self.metrics.dict(),
            "uptime": time.time() - self.start_time
        }
        
        return AgentMessage(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type="status_response",
            content=status_info,
            correlation_id=message.correlation_id
        )
    
    async def _handle_health_check(self, message: AgentMessage) -> AgentMessage:
        """Handle health check requests"""
        is_healthy = (
            self.status != AgentStatus.ERROR and
            self.last_error is None and
            self.metrics.success_rate > 0.5
        )
        
        health_info = {
            "agent_id": self.agent_id,
            "healthy": is_healthy,
            "status": self.status.value,
            "last_error": str(self.last_error) if self.last_error else None,
            "success_rate": self.metrics.success_rate
        }
        
        return AgentMessage(
            sender_id=self.agent_id,
            recipient_id=message.sender_id,
            message_type="health_response",
            content=health_info,
            correlation_id=message.correlation_id
        )
    
    def add_memory(self, key: str, value: Any, ttl: Optional[int] = None):
        """Add information to agent's memory"""
        self.memory[key] = {
            "value": value,
            "timestamp": datetime.utcnow(),
            "ttl": ttl
        }
        
        # Add to context window
        self.context_window.append({
            "type": "memory_update",
            "key": key,
            "timestamp": datetime.utcnow()
        })
        
        # Maintain context window size
        if len(self.context_window) > self.max_context_size:
            self.context_window.pop(0)
    
    def get_memory(self, key: str) -> Optional[Any]:
        """Retrieve information from agent's memory"""
        if key in self.memory:
            memory_item = self.memory[key]
            
            # Check TTL
            if memory_item.get("ttl"):
                age = (datetime.utcnow() - memory_item["timestamp"]).seconds
                if age > memory_item["ttl"]:
                    del self.memory[key]
                    return None
            
            return memory_item["value"]
        
        return None
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get comprehensive agent information"""
        return {
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

# Utility function to create agents with common configuration
def create_agent(
    agent_class,
    agent_id: str,
    name: str,
    role: str,
    capabilities: List[AgentCapability],
    llm_service: LLMService
):
    """Factory function to create agents with standard configuration"""
    return agent_class(
        agent_id=agent_id,
        name=name,
        role=role,
        llm_service=llm_service,
        capabilities=capabilities
    )
