import logging
import sys
from datetime import datetime
from typing import Any, Optional
import structlog
from rich.console import Console
from rich.logging import RichHandler
from rich.traceback import install
from pathlib import Path
import json

from app.core.config import settings

# Install rich traceback for better error display
install(show_locals=True)

# Rich console for beautiful output
console = Console()

def setup_logging() -> logging.Logger:
    """
    Configure comprehensive logging system with both structured logging
    and rich formatting for development and production environments.
    """
    
    # Configure structlog processors
    processors = [
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
    ]
    
    # Add appropriate renderer based on environment
    if settings.DEBUG and settings.ENABLE_RICH_LOGGING:
        processors.append(
            structlog.dev.ConsoleRenderer(colors=True, exception_formatter=structlog.dev.plain_traceback)
        )
    else:
        processors.append(structlog.processors.JSONRenderer())
    
    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Configure handlers
    handlers = []
    
    # Console handler
    if settings.ENABLE_RICH_LOGGING and settings.DEBUG:
        handlers.append(
            RichHandler(
                console=console,
                rich_tracebacks=True,
                tracebacks_show_locals=True,
                level=getattr(logging, settings.LOG_LEVEL)
            )
        )
    else:
        handlers.append(
            logging.StreamHandler(sys.stdout)
        )
    
    # File handler (if specified)
    if settings.LOG_FILE:
        log_file = Path(settings.LOG_FILE)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.FileHandler(
                settings.LOG_FILE,
                encoding='utf-8'
            )
        )
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL),
        format=settings.LOG_FORMAT,
        handlers=handlers,
        force=True
    )
    
    # Silence noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    
    # Create and return application logger
    logger = structlog.get_logger("ai_content_generator")
    logger.info("Logging system initialized", log_level=settings.LOG_LEVEL)
    
    return logger

class AgentLogger:
    """
    Specialized logger for multi-agent workflows with agent-specific context.
    Provides structured logging with agent identification and performance tracking.
    """
    
    def __init__(self, agent_id: str, agent_name: str):
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.logger = structlog.get_logger("agent").bind(
            agent_id=agent_id,
            agent_name=agent_name,
            component="agent"
        )
    
    def info(self, message: str, **kwargs):
        """Log informational message with agent context"""
        self.logger.info(message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message with agent context"""
        self.logger.warning(message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message with agent context"""
        self.logger.error(message, **kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log debug message with agent context"""
        self.logger.debug(message, **kwargs)
    
    def task_start(self, task_id: str, task_type: str, **context):
        """Log task initiation"""
        self.logger.info(
            "Agent task started",
            task_id=task_id,
            task_type=task_type,
            timestamp=datetime.utcnow().isoformat(),
            **context
        )
    
    def task_complete(self, task_id: str, duration: float, success: bool, **context):
        """Log task completion with performance metrics"""
        self.logger.info(
            "Agent task completed",
            task_id=task_id,
            duration_seconds=round(duration, 3),
            success=success,
            timestamp=datetime.utcnow().isoformat(),
            **context
        )
    
    def task_error(self, task_id: str, error: Exception, **context):
        """Log task error with detailed context"""
        self.logger.error(
            "Agent task failed",
            task_id=task_id,
            error_type=type(error).__name__,
            error_message=str(error),
            timestamp=datetime.utcnow().isoformat(),
            **context,
            exc_info=True
        )
    
    def workflow_event(self, event_type: str, **context):
        """Log workflow-specific events"""
        self.logger.info(
            f"Workflow event: {event_type}",
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat(),
            **context
        )

class WorkflowLogger:
    """
    Logger specialized for tracking multi-agent workflow execution.
    Provides comprehensive workflow state and performance logging.
    """
    
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.logger = structlog.get_logger("workflow").bind(
            workflow_id=workflow_id,
            component="workflow"
        )
    
    def workflow_start(self, workflow_type: str, agents: list, **context):
        """Log workflow initiation"""
        self.logger.info(
            "Multi-agent workflow started",
            workflow_type=workflow_type,
            agents_count=len(agents),
            agents=agents,
            timestamp=datetime.utcnow().isoformat(),
            **context
        )
    
    def workflow_step(self, step_name: str, agent_id: str, status: str, **context):
        """Log individual workflow step"""
        self.logger.info(
            f"Workflow step: {step_name}",
            step_name=step_name,
            agent_id=agent_id,
            status=status,
            timestamp=datetime.utcnow().isoformat(),
            **context
        )
    
    def workflow_complete(self, success: bool, total_duration: float, results: dict, **context):
        """Log workflow completion with full results"""
        self.logger.info(
            "Multi-agent workflow completed",
            success=success,
            total_duration_seconds=round(total_duration, 3),
            results_summary=self._summarize_results(results),
            timestamp=datetime.utcnow().isoformat(),
            **context
        )
    
    def workflow_error(self, error: Exception, step_name: Optional[str] = None, **context):
        """Log workflow error with detailed context"""
        self.logger.error(
            "Workflow execution failed",
            step_name=step_name,
            error_type=type(error).__name__,
            error_message=str(error),
            timestamp=datetime.utcnow().isoformat(),
            **context,
            exc_info=True
        )
    
    def agent_communication(self, sender: str, receiver: str, message_type: str, **context):
        """Log inter-agent communication"""
        self.logger.debug(
            "Agent communication",
            sender=sender,
            receiver=receiver,
            message_type=message_type,
            timestamp=datetime.utcnow().isoformat(),
            **context
        )
    
    def _summarize_results(self, results: dict) -> dict:
        """Create a summary of workflow results for logging"""
        return {
            "content_length": len(str(results.get("final_content", ""))),
            "quality_score": results.get("quality_score", 0),
            "agents_involved": len(results.get("agent_contributions", [])),
            "persona_alignment": results.get("persona_alignment", 0)
        }

# Initialize global logger
logger = setup_logging()

# Export convenience function for getting agent loggers
def get_agent_logger(agent_id: str, agent_name: str) -> AgentLogger:
    """Get a configured agent logger"""
    return AgentLogger(agent_id, agent_name)

def get_workflow_logger(workflow_id: str) -> WorkflowLogger:
    """Get a configured workflow logger"""
    return WorkflowLogger(workflow_id)
