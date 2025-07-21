import pytest
import asyncio
from typing import Dict, Any
from unittest.mock import Mock, AsyncMock
import os
import tempfile

from app.core.config import settings
from app.services.llm_service import LLMService, GroqProvider
from app.agents.base_agent import BaseAgent
from app.models.agent_schemas import AgentCapability

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_groq_provider():
    """Mock Groq provider for testing without API calls."""
    provider = Mock(spec=GroqProvider)
    provider.generate = AsyncMock(return_value=Mock(
        content="Test generated content",
        model_used="llama-3.1-70b-versatile",
        generation_time=1.5,
        tokens_used=150,
        tokens_per_second=100.0,
        confidence_estimate=0.85,
        completion_reason="stop",
        request_id="test_request_123",
        estimated_cost=0.01
    ))
    provider.health_check = AsyncMock(return_value=True)
    provider.get_stats = Mock(return_value={
        "provider": "groq",
        "total_requests": 10,
        "total_tokens": 1500,
        "total_cost": 0.10
    })
    return provider

@pytest.fixture
def mock_llm_service(mock_groq_provider):
    """Mock LLM service with mocked provider."""
    service = Mock(spec=LLMService)
    service.providers = {"groq": mock_groq_provider}
    service.default_provider = "groq"
    service.generate_content = AsyncMock(return_value=Mock(
        content="Test generated content",
        model_used="llama-3.1-70b-versatile",
        generation_time=1.5,
        tokens_used=150
    ))
    service.health_check = AsyncMock(return_value={"groq": True})
    service.get_service_stats = Mock(return_value={
        "providers": {"groq": mock_groq_provider.get_stats()},
        "cache": {"hits": 5, "misses": 5, "hit_rate": 0.5},
        "total_providers": 1
    })
    return service

@pytest.fixture
def sample_persona():
    """Sample persona data for testing."""
    return {
        "id": "young_parent",
        "name": "Young Parent",
        "type": "young_parent",
        "demographics": {
            "age_range": "28-35",
            "income": "$50k-75k",
            "location": "Suburban",
            "education": "College graduate"
        },
        "pain_points": [
            "Limited time for research",
            "Budget constraints",
            "Safety concerns for children"
        ],
        "goals": [
            "Provide best for children",
            "Save time and money",
            "Maintain family health"
        ],
        "communication_preferences": {
            "tone": "warm, empathetic",
            "formality": "casual but respectful",
            "terminology": "simple, avoid jargon"
        },
        "content_preferences": {
            "length": "concise, scannable",
            "format": "bullet points, short paragraphs"
        }
    }

@pytest.fixture
def sample_content_request():
    """Sample content generation request for testing."""
    return {
        "persona_id": "young_parent",
        "content_type": "ad",
        "topic": "Organic baby food",
        "word_count": 150,
        "creativity_level": 0.7,
        "additional_context": "Focus on safety and nutrition",
        "enable_qa": True,
        "enable_variations": False,
        "enable_realtime_updates": False
    }

@pytest.fixture
def sample_agent_capabilities():
    """Sample agent capabilities for testing."""
    return [
        AgentCapability(
            name="test_capability",
            description="Test capability for unit testing",
            proficiency_level=0.9,
            required_resources=["groq_api"],
            average_execution_time=10.0
        )
    ]

@pytest.fixture
def temp_config_file():
    """Temporary configuration file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.env', delete=False) as f:
        f.write("""
        GROQ_API_KEY=test_api_key
        DEBUG=True
        LOG_LEVEL=DEBUG
        MAX_CONCURRENT_AGENTS=3
        """)
        temp_file = f.name
    
    yield temp_file
    
    # Cleanup
    os.unlink(temp_file)

@pytest.fixture
def mock_workflow_state():
    """Mock workflow state for testing."""
    from app.models.workflow_schemas import WorkflowState, WorkflowStatus
    from datetime import datetime
    
    return WorkflowState(
        workflow_id="test_workflow_123",
        workflow_name="test_content_generation",
        workflow_type="standard",
        request_id="test_request_123",
        original_request={"content_type": "ad", "topic": "test product"},
        status=WorkflowStatus.RUNNING,
        current_step="persona_research",
        agent_responses={},
        workflow_steps=[],
        completed_steps=[],
        communications=[],
        created_at=datetime.utcnow(),
        agent_logs=[],
        errors=[],
        warnings=[],
        metadata={}
    )
