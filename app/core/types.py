"""Type definitions for the multi-agent system"""
from typing import Dict, Any, List, Optional, Union, TypeVar, Protocol
from enum import Enum

# Type aliases for better readability
AgentID = str
TaskID = str
WorkflowID = str
PersonaID = str

# Generic types
T = TypeVar('T')
AgentResponseType = TypeVar('AgentResponseType')

class Processable(Protocol):
    """Protocol for objects that can be processed by agents"""
    def validate(self) -> bool: ...
    def to_dict(self) -> Dict[str, Any]: ...

# Configuration types
AgentConfig = Dict[str, Any]
WorkflowConfig = Dict[str, Any]
