"""Enhanced configuration management"""
from pydantic import BaseSettings, Field
from typing import List, Dict, Any, Optional

class AgentSettings(BaseSettings):
    """Agent-specific configuration"""
    max_retry_attempts: int = Field(3, description="Maximum task retry attempts")
    default_timeout: int = Field(60, description="Default task timeout in seconds")
    confidence_threshold: float = Field(0.7, description="Minimum confidence threshold")

class WorkflowSettings(BaseSettings):
    """Workflow configuration"""
    max_concurrent_workflows: int = Field(5, description="Maximum concurrent workflows")
    workflow_timeout: int = Field(300, description="Workflow timeout in seconds")
    enable_caching: bool = Field(True, description="Enable workflow result caching")

class ContentSettings(BaseSettings):
    """Content generation settings"""
    supported_content_types: List[str] = Field(
        default=['ad', 'landing_page', 'blog_intro', 'email', 'social_media'],
        description="Supported content types"
    )
    word_count_limits: Dict[str, tuple] = Field(
        default={
            'ad': (25, 100),
            'landing_page': (150, 500),
            'email': (75, 300)
        },
        description="Word count limits by content type"
    )

class Settings(BaseSettings):
    """Main application settings"""
    # Existing settings...
    
    # Enhanced settings
    agent: AgentSettings = AgentSettings()
    workflow: WorkflowSettings = WorkflowSettings()
    content: ContentSettings = ContentSettings()
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
