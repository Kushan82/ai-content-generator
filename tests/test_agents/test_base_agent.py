# tests/unit/test_agents/test_base_agent.py
import pytest
from unittest.mock import Mock, AsyncMock
from app.agents.base_agent import BaseAgent
from app.core.exceptions import AgentError

class TestBaseAgent:
    """Unit tests for BaseAgent functionality"""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock agent for testing"""
        class MockAgent(BaseAgent):
            async def process_task(self, task):
                return {"result": "test"}
        
        return MockAgent("test_agent", "Test Agent", "Testing", Mock())
    
    @pytest.mark.asyncio
    async def test_task_execution_success(self, mock_agent):
        """Test successful task execution"""
        task = Mock()
        task.input_data = {"test": "data"}
        
        result = await mock_agent.execute_task(task)
        
        assert result is not None
        assert mock_agent.status == AgentStatus.IDLE
    
    @pytest.mark.asyncio 
    async def test_task_execution_failure(self, mock_agent):
        """Test task execution failure handling"""
        task = Mock()
        task.input_data = None  # This should cause validation failure
        
        with pytest.raises(AgentError):
            await mock_agent.execute_task(task)
