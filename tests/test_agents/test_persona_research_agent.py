import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from app.agents.persona_research_agent import PersonaResearchAgent
from app.agents.base_agent import TaskRequest
from app.models.agent_schemas import AgentStatus

class TestPersonaResearchAgent:
    """
    Comprehensive test suite for the Persona Research Agent.
    Tests agent initialization, task processing, error handling, and performance.
    """
    
    @pytest.fixture
    async def persona_agent(self, mock_llm_service):
        """Create persona research agent instance for testing."""
        return PersonaResearchAgent(mock_llm_service)
    
    @pytest.mark.asyncio
    async def test_agent_initialization(self, persona_agent):
        """Test agent initializes correctly with proper capabilities."""
        assert persona_agent.agent_id == "persona_researcher"
        assert persona_agent.name == "Persona Research Specialist"
        assert persona_agent.role == "Market Research & Demographic Analysis"
        assert len(persona_agent.capabilities) == 4
        assert persona_agent.status == AgentStatus.IDLE
        
        # Test capability names
        capability_names = [cap.name for cap in persona_agent.capabilities]
        expected_capabilities = [
            "demographic_analysis",
            "psychographic_profiling", 
            "pain_point_analysis",
            "competitive_research"
        ]
        assert all(cap in capability_names for cap in expected_capabilities)
    
    @pytest.mark.asyncio
    async def test_successful_persona_research(self, persona_agent, sample_persona):
        """Test successful persona research execution."""
        
        # Create test task
        task = TaskRequest(
            task_type="persona_research",
            input_data={
                "persona_id": "young_parent",
                "additional_context": "Focus on family safety",
                "research_depth": "comprehensive"
            }
        )
        
        # Mock LLM response
        mock_research_data = {
            "demographics": sample_persona["demographics"],
            "psychographics": sample_persona["pain_points"],
            "pain_points": sample_persona["pain_points"],
            "goals": sample_persona["goals"],
            "key_insights": ["Insight 1", "Insight 2"]
        }
        
        persona_agent.llm_service.generate_content.return_value.content = str(mock_research_data)
        
        # Execute task
        result = await persona_agent.execute_task(task)
        
        # Verify results
        assert result.status == AgentStatus.COMPLETED
        assert result.error_message is None
        assert "persona_research" in result.content
        assert result.confidence_score > 0.0
        assert result.processing_time > 0.0
        
        # Verify LLM service was called
        persona_agent.llm_service.generate_content.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_missing_persona_id_error(self, persona_agent):
        """Test error handling when persona_id is missing."""
        
        task = TaskRequest(
            task_type="persona_research",
            input_data={
                "additional_context": "Test context"
                # Missing persona_id
            }
        )
        
        result = await persona_agent.execute_task(task)
        
        assert result.status == AgentStatus.ERROR
        assert result.error_message is not None
        assert "persona_id is required" in result.error_message
    
    @pytest.mark.asyncio
    async def test_llm_service_timeout(self, persona_agent):
        """Test handling of LLM service timeout."""
        
        # Configure mock to raise timeout
        persona_agent.llm_service.generate_content.side_effect = asyncio.TimeoutError("Request timed out")
        
        task = TaskRequest(
            task_type="persona_research",
            input_data={"persona_id": "young_parent"},
            timeout=5  # Short timeout
        )
        
        result = await persona_agent.execute_task(task)
        
        assert result.status == AgentStatus.TIMEOUT
        assert "timed out" in result.error_message.lower()
    
    @pytest.mark.asyncio
    async def test_confidence_score_calculation(self, persona_agent):
        """Test confidence score calculation based on response quality."""
        
        task = TaskRequest(
            task_type="persona_research",
            input_data={"persona_id": "young_parent"}
        )
        
        # Mock high-quality response
        persona_agent.llm_service.generate_content.return_value.content = """
        {
            "demographics": {"age_range": "28-35", "income": "$50k-75k"},
            "pain_points": ["time constraints", "budget concerns"],
            "goals": ["family wellness", "convenience"],
            "key_insights": ["detailed insight 1", "detailed insight 2"]
        }
        """
        
        result = await persona_agent.execute_task(task)
        
        # High-quality response should have good confidence
        assert result.confidence_score >= 0.7
        assert result.status == AgentStatus.COMPLETED
    
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self, persona_agent):
        """Test that agent properly tracks performance metrics."""
        
        initial_completed = persona_agent.metrics.total_tasks_completed
        initial_success_rate = persona_agent.metrics.success_rate
        
        task = TaskRequest(
            task_type="persona_research",
            input_data={"persona_id": "young_parent"}
        )
        
        persona_agent.llm_service.generate_content.return_value.content = '{"demographics": {}}'
        
        await persona_agent.execute_task(task)
        
        # Verify metrics updated
        assert persona_agent.metrics.total_tasks_completed == initial_completed + 1
        assert persona_agent.metrics.average_response_time > 0
        assert persona_agent.metrics.success_rate >= initial_success_rate
    
    @pytest.mark.asyncio
    async def test_memory_storage(self, persona_agent):
        """Test that agent stores research results in memory."""
        
        task = TaskRequest(
            task_type="persona_research",
            input_data={"persona_id": "young_parent"}
        )
        
        persona_agent.llm_service.generate_content.return_value.content = '{"demographics": {"test": "data"}}'
        
        await persona_agent.execute_task(task)
        
        # Check that research was stored in memory
        stored_research = persona_agent.get_memory("persona_research_young_parent")
        assert stored_research is not None
