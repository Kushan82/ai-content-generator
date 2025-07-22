"""Custom exceptions for multi-agent system"""

class MultiAgentError(Exception):
    """Base exception for multi-agent system"""
    def __init__(self, message: str, error_code: str = None, details: dict = None):
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)

class AgentError(MultiAgentError):
    """Agent-specific errors"""
    pass

class WorkflowError(MultiAgentError):
    """Workflow execution errors"""
    pass

class PersonaNotFoundError(MultiAgentError):
    """Persona not found error"""
    pass

class ContentGenerationError(MultiAgentError):
    """Content generation errors"""
    pass
