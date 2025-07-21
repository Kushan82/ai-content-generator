import pytest
import asyncio
from unittest.mock import Mock, AsyncMock

from app.services.workflow_engine import WorkflowEngine
from app.models.workflow_schemas import WorkflowState, WorkflowStatus
from app.agents.base_agent import BaseAgent

class TestWorkflowEngine:
    """
    Integration tests for the LangGraph-powered workflow engine.
    Tests workflow execution, state management, and error recovery.
    """
    
    @pytest.fixture
    def mock_agents(self, mock_llm_service, sample_agent_capabilities):
        """Create mock agents for workflow testing."""
        agents = {}
        
        for agent_id in ["persona_researcher", "content_strategist", "creative_generator", "quality_assurance"]:
            agent = Mock(spec=BaseAgent)
            agent.agent_id = agent_id
            agent.name = f"Test {agent_id.replace('_', ' ').title()}"
            agent.capabilities = sample_agent_capabilities
            agent.execute_task = AsyncMock(return_value=Mock(
                status="completed",
                content={"test_output": f"Result from {agent_id}"},
                confidence_score=0.85,
                processing_time=1.0,
                tokens_used=100,
                api_calls_made=1
            ))
            agents[agent_id] = agent
        
        return agents
    
    @pytest.fixture
    def workflow_engine(self, mock_agents):
        """Create workflow engine with mock agents."""
        return WorkflowEngine(mock_agents)
    
    @pytest.mark.asyncio
    async def test_workflow_initialization(self, workflow_engine):
        """Test workflow engine initializes correctly."""
        assert workflow_engine.agents is not None
        assert len(workflow_engine.agents) == 4
        assert workflow_engine.workflow_graph is not None
        assert workflow_engine.checkpointer is not None
    
    @pytest.mark.asyncio
    async def test_successful_workflow_execution(self, workflow_engine, sample_content_request):
        """Test complete successful workflow execution."""
        
        # Execute workflow
        result = await workflow_engine.execute_workflow(sample_content_request)
        
        # Verify workflow completed successfully
        assert isinstance(result, WorkflowState)
        assert result.status == WorkflowStatus.COMPLETED
        assert result.workflow_id is not None
        assert result.completed_at is not None
        assert len(result.errors) == 0
        
        # Verify agent responses
        assert len(result.agent_responses) >= 3  # At least persona, strategy, generation
        
        # Verify final content exists
        assert result.final_content is not None
        assert len(result.final_content) > 0
    
    @pytest.mark.asyncio
    async def test_workflow_with_qa_disabled(self, workflow_engine, sample_content_request):
        """Test workflow execution with QA step disabled."""
        
        # Disable QA in request
        request_with_no_qa = sample_content_request.copy()
        request_with_no_qa["skip_qa"] = True
        
        result = await workflow_engine.execute_workflow(request_with_no_qa)
        
        assert result.status == WorkflowStatus.COMPLETED
        # QA agent should not be in responses when skipped
        assert "quality_assurance" not in result.agent_responses or \
               result.agent_responses.get("quality_assurance") is None
    
    @pytest.mark.asyncio
    async def test_workflow_error_handling(self, workflow_engine, mock_agents, sample_content_request):
        """Test workflow handles agent failures gracefully."""
        
        # Make persona research agent fail
        mock_agents["persona_researcher"].execute_task.side_effect = Exception("Simulated agent failure")
        
        # Execute workflow
        with pytest.raises(Exception):
            await workflow_engine.execute_workflow(sample_content_request)
        
        # Verify error was recorded
        # Note: In a real implementation, you might want graceful degradation
        # instead of complete failure
    
    @pytest.mark.asyncio
    async def test_workflow_state_tracking(self, workflow_engine, sample_content_request):
        """Test that workflow state is properly tracked throughout execution."""
        
        # Start workflow execution
        result = await workflow_engine.execute_workflow(sample_content_request)
        
        # Verify state progression was tracked
        assert result.started_at is not None
        assert result.completed_at is not None
        assert result.total_duration_seconds is not None
        assert result.total_duration_seconds > 0
        
        # Verify step timings were recorded
        assert len(result.step_timings) > 0
    
    @pytest.mark.asyncio
    async def test_workflow_cancellation(self, workflow_engine):
        """Test workflow cancellation functionality."""
        
        workflow_id = "test_workflow_123"
        
        # Cancel workflow (even if not running)
        cancelled = await workflow_engine.cancel_workflow(workflow_id)
        
        # Should return False for non-existent workflow
        assert cancelled == False
    
    @pytest.mark.asyncio
    async def test_workflow_status_retrieval(self, workflow_engine):
        """Test workflow status retrieval."""
        
        workflow_id = "non_existent_workflow"
        
        status = await workflow_engine.get_workflow_status(workflow_id)
        
        # Should return None for non-existent workflow
        assert status is None
    
    @pytest.mark.asyncio
    async def test_workflow_progress_calculation(self, workflow_engine):
        """Test workflow progress calculation."""
        
        # Test progress calculation method
        from app.models.workflow_schemas import WorkflowState
        
        test_workflow = WorkflowState(
            workflow_id="test",
            workflow_name="test",
            workflow_type="test",
            request_id="test",
            original_request={},
            current_step="content_strategy"
        )
        
        progress = workflow_engine._calculate_progress(test_workflow)
        
        assert 0.0 <= progress <= 1.0
        assert progress > 0.0  # Should have made some progress
