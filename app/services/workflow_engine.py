import asyncio
from typing import Dict, Any, List, Optional, Callable
from datetime import datetime
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from app.models.workflow_schemas import WorkflowState, WorkflowStatus
from app.agents.base_agent import BaseAgent
from app.core.logging import get_workflow_logger, WorkflowLogger
from app.core.config import settings

class WorkflowEngine:
    """
    LangGraph-powered workflow engine for orchestrating multi-agent content generation.
    
    This engine manages complex workflows using LangGraph's state management,
    provides checkpointing for recovery, and enables sophisticated agent coordination
    patterns including conditional flows and iterative refinement.
    """
    
    def __init__(self, agents: Dict[str, BaseAgent]):
        self.agents = agents
        self.checkpointer = MemorySaver()
        self.active_workflows: Dict[str, WorkflowState] = {}
        
        # Build the LangGraph workflow
        self.workflow_graph = self._build_workflow_graph()
        
        self.logger = get_workflow_logger("workflow_engine")
    
    def _build_workflow_graph(self) -> StateGraph:
        """
        Build the LangGraph state machine for multi-agent content generation workflow.
        """
        # Define the workflow state structure
        workflow = StateGraph(WorkflowState)
        
        # Add workflow nodes (agent execution steps)
        workflow.add_node("start_workflow", self._start_workflow_step)
        workflow.add_node("persona_research", self._persona_research_step)
        workflow.add_node("content_strategy", self._content_strategy_step)
        workflow.add_node("creative_generation", self._creative_generation_step)
        workflow.add_node("quality_assurance", self._quality_assurance_step)
        workflow.add_node("workflow_synthesis", self._workflow_synthesis_step)
        workflow.add_node("finalize_workflow", self._finalize_workflow_step)
        
        # Define conditional routing logic
        workflow.add_conditional_edges(
            "start_workflow",
            self._route_from_start,
            {
                "persona_research": "persona_research",
                "content_strategy": "content_strategy",
                "end": END
            }
        )
        
        # Define sequential workflow edges
        workflow.add_edge("persona_research", "content_strategy")
        workflow.add_edge("content_strategy", "creative_generation")
        
        # Add conditional QA routing
        workflow.add_conditional_edges(
            "creative_generation",
            self._route_after_generation,
            {
                "quality_assurance": "quality_assurance",
                "synthesis": "workflow_synthesis"
            }
        )
        
        workflow.add_edge("quality_assurance", "workflow_synthesis")
        workflow.add_edge("workflow_synthesis", "finalize_workflow")
        workflow.add_edge("finalize_workflow", END)
        
        # Set entry point
        workflow.set_entry_point("start_workflow")
        
        return workflow.compile(checkpointer=self.checkpointer)
    
    async def execute_workflow(self, request: Dict[str, Any]) -> WorkflowState:
        """
        Execute a complete multi-agent workflow using LangGraph orchestration.
        """
        workflow_id = str(uuid.uuid4())
        
        # Initialize workflow state
        initial_state = WorkflowState(
            workflow_id=workflow_id,
            workflow_name="content_generation",
            workflow_type="multi_agent_content_generation",
            request_id=request.get("request_id", workflow_id),
            original_request=request,
            status=WorkflowStatus.INITIALIZING
        )
        
        self.active_workflows[workflow_id] = initial_state
        
        try:
            # Execute the workflow graph
            config = {"configurable": {"thread_id": workflow_id}}
            
            final_state = await self.workflow_graph.ainvoke(
                initial_state.dict(),
                config=config
            )
            
            # Convert back to WorkflowState object
            workflow_state = WorkflowState(**final_state)
            workflow_state.status = WorkflowStatus.COMPLETED
            workflow_state.completed_at = datetime.utcnow()
            workflow_state.calculate_total_duration()
            
            self.active_workflows[workflow_id] = workflow_state
            
            return workflow_state
            
        except Exception as e:
            # Handle workflow failure
            failed_state = self.active_workflows[workflow_id]
            failed_state.status = WorkflowStatus.FAILED
            failed_state.errors.append(str(e))
            failed_state.completed_at = datetime.utcnow()
            
            self.logger.workflow_error(e, workflow_id=workflow_id)
            raise
    
    async def _start_workflow_step(self, state: WorkflowState) -> WorkflowState:
        """Initialize and validate workflow execution."""
        state["status"] = WorkflowStatus.RUNNING.value
        state["started_at"] = datetime.utcnow()
        state["current_step"] = "initialization"
        
        # Validate required inputs
        request = state["original_request"]
        required_fields = ["content_type", "topic"]
        
        for field in required_fields:
            if field not in request:
                state["errors"].append(f"Missing required field: {field}")
                state["status"] = WorkflowStatus.FAILED.value
                return state
        
        # Log workflow start
        self.logger.workflow_start(
            workflow_type="content_generation",
            agents=list(self.agents.keys()),
            workflow_id=state["workflow_id"]
        )
        
        return state
    
    async def _persona_research_step(self, state: WorkflowState) -> WorkflowState:
        """Execute persona research agent step."""
        if "persona_researcher" not in self.agents:
            state["errors"].append("Persona research agent not available")
            return state
        
        state["current_step"] = "persona_research"
        step_start_time = datetime.utcnow()
        
        try:
            agent = self.agents["persona_researcher"]
            
            # Create task for persona research
            from app.agents.base_agent import TaskRequest
            task = TaskRequest(
                task_type="persona_research",
                input_data={
                    "persona_id": state["original_request"].get("persona_id"),
                    "additional_context": state["original_request"].get("additional_context"),
                    "research_depth": state["original_request"].get("research_depth", "comprehensive")
                },
                workflow_id=state["workflow_id"]
            )
            
            # Execute agent task
            result = await agent.execute_task(task)
            
            # Store result in workflow state
            state["agent_responses"]["persona_researcher"] = result
            state["persona_research"] = result.content
            
            # Update timing
            step_end_time = datetime.utcnow()
            state["step_timings"]["persona_research"] = (step_end_time - step_start_time).total_seconds()
            
            self.logger.workflow_step(
                step_name="persona_research",
                agent_id="persona_researcher",
                status="completed",
                workflow_id=state["workflow_id"]
            )
            
        except Exception as e:
            state["errors"].append(f"Persona research failed: {str(e)}")
            self.logger.workflow_error(e, step_name="persona_research")
        
        return state
    
    async def _content_strategy_step(self, state: WorkflowState) -> WorkflowState:
        """Execute content strategy agent step."""
        if "content_strategist" not in self.agents:
            state["errors"].append("Content strategy agent not available")
            return state
        
        state["current_step"] = "content_strategy"
        step_start_time = datetime.utcnow()
        
        try:
            agent = self.agents["content_strategist"]
            
            # Create task with persona research data
            from app.agents.base_agent import TaskRequest
            task = TaskRequest(
                task_type="strategy_development",
                input_data={
                    "persona_research": state.get("persona_research", {}),
                    "content_type": state["original_request"]["content_type"],
                    "topic": state["original_request"]["topic"],
                    "additional_context": state["original_request"].get("additional_context"),
                    "business_objectives": state["original_request"].get("business_objectives", [])
                },
                workflow_id=state["workflow_id"]
            )
            
            # Execute agent task
            result = await agent.execute_task(task)
            
            # Store result
            state["agent_responses"]["content_strategist"] = result
            state["content_strategy"] = result.content
            
            # Update timing
            step_end_time = datetime.utcnow()
            state["step_timings"]["content_strategy"] = (step_end_time - step_start_time).total_seconds()
            
            self.logger.workflow_step(
                step_name="content_strategy",
                agent_id="content_strategist", 
                status="completed",
                workflow_id=state["workflow_id"]
            )
            
        except Exception as e:
            state["errors"].append(f"Content strategy failed: {str(e)}")
            self.logger.workflow_error(e, step_name="content_strategy")
        
        return state
    
    async def _creative_generation_step(self, state: WorkflowState) -> WorkflowState:
        """Execute creative generation agent step."""
        if "creative_generator" not in self.agents:
            state["errors"].append("Creative generation agent not available")
            return state
        
        state["current_step"] = "creative_generation"
        step_start_time = datetime.utcnow()
        
        try:
            agent = self.agents["creative_generator"]
            
            # Create comprehensive task with all previous data
            from app.agents.base_agent import TaskRequest
            task = TaskRequest(
                task_type="content_generation",
                input_data={
                    "persona_research": state.get("persona_research", {}),
                    "content_strategy": state.get("content_strategy", {}),
                    "content_type": state["original_request"]["content_type"],
                    "topic": state["original_request"]["topic"],
                    "word_count": state["original_request"].get("word_count", 200),
                    "creativity_level": state["original_request"].get("creativity_level", 0.7),
                    "brand_voice": state["original_request"].get("brand_voice", {})
                },
                workflow_id=state["workflow_id"]
            )
            
            # Execute agent task
            result = await agent.execute_task(task)
            
            # Store result
            state["agent_responses"]["creative_generator"] = result
            state["generated_content"] = result.content.get("generated_content", "")
            
            # Update timing
            step_end_time = datetime.utcnow()
            state["step_timings"]["creative_generation"] = (step_end_time - step_start_time).total_seconds()
            
            self.logger.workflow_step(
                step_name="creative_generation",
                agent_id="creative_generator",
                status="completed", 
                workflow_id=state["workflow_id"]
            )
            
        except Exception as e:
            state["errors"].append(f"Creative generation failed: {str(e)}")
            self.logger.workflow_error(e, step_name="creative_generation")
        
        return state
    
    async def _quality_assurance_step(self, state: WorkflowState) -> WorkflowState:
        """Execute quality assurance agent step."""
        if "quality_assurance" not in self.agents:
            # QA is optional, continue without it
            return state
        
        state["current_step"] = "quality_assurance"
        step_start_time = datetime.utcnow()
        
        try:
            agent = self.agents["quality_assurance"]
            
            # Create QA task with all relevant data
            from app.agents.base_agent import TaskRequest
            task = TaskRequest(
                task_type="quality_assessment",
                input_data={
                    "generated_content": state.get("generated_content", ""),
                    "content_type": state["original_request"]["content_type"],
                    "persona_research": state.get("persona_research", {}),
                    "content_strategy": state.get("content_strategy", {}),
                    "brand_guidelines": state["original_request"].get("brand_guidelines", {})
                },
                workflow_id=state["workflow_id"]
            )
            
            # Execute QA
            result = await agent.execute_task(task)
            
            # Store QA results
            state["agent_responses"]["quality_assurance"] = result
            state["qa_feedback"] = result.content
            
            # Use optimized content if available
            if result.content.get("optimized_content"):
                state["generated_content"] = result.content["optimized_content"]
            
            # Update timing
            step_end_time = datetime.utcnow()
            state["step_timings"]["quality_assurance"] = (step_end_time - step_start_time).total_seconds()
            
            self.logger.workflow_step(
                step_name="quality_assurance",
                agent_id="quality_assurance",
                status="completed",
                workflow_id=state["workflow_id"]
            )
            
        except Exception as e:
            state["errors"].append(f"Quality assurance failed: {str(e)}")
            self.logger.workflow_error(e, step_name="quality_assurance")
        
        return state
    
    async def _workflow_synthesis_step(self, state: WorkflowState) -> WorkflowState:
        """Synthesize results from all agents into final output."""
        state["current_step"] = "synthesis"
        
        try:
            # Calculate overall quality score
            qa_feedback = state.get("qa_feedback", {})
            quality_assessment = qa_feedback.get("quality_assessment", {})
            state["overall_quality_score"] = quality_assessment.get("overall_score", 0.8)
            
            # Calculate overall confidence
            confidence_scores = []
            for agent_response in state["agent_responses"].values():
                if hasattr(agent_response, 'confidence_score'):
                    confidence_scores.append(agent_response.confidence_score)
                elif isinstance(agent_response, dict) and 'confidence_score' in agent_response:
                    confidence_scores.append(agent_response['confidence_score'])
            
            if confidence_scores:
                state["overall_confidence_score"] = sum(confidence_scores) / len(confidence_scores)
            
            # Aggregate token usage and API calls
            for agent_response in state["agent_responses"].values():
                if hasattr(agent_response, 'tokens_used'):
                    state["total_tokens_used"] += agent_response.tokens_used
                elif isinstance(agent_response, dict) and 'tokens_used' in agent_response:
                    state["total_tokens_used"] += agent_response['tokens_used']
                    
                if hasattr(agent_response, 'api_calls_made'):
                    state["total_api_calls"] += agent_response.api_calls_made
                elif isinstance(agent_response, dict) and 'api_calls' in agent_response:
                    state["total_api_calls"] += agent_response['api_calls']
            
            # Set final content
            state["final_content"] = state.get("generated_content", "")
            
        except Exception as e:
            state["errors"].append(f"Workflow synthesis failed: {str(e)}")
            self.logger.workflow_error(e, step_name="synthesis")
        
        return state
    
    async def _finalize_workflow_step(self, state: WorkflowState) -> WorkflowState:
        """Finalize workflow execution and cleanup."""
        state["current_step"] = "finalization"
        state["workflow_status"] = "completed"
        
        # Log workflow completion
        self.logger.workflow_complete(
            success=len(state["errors"]) == 0,
            total_duration=state.get("total_processing_time", 0),
            results={"final_content_length": len(state.get("final_content", ""))},
            workflow_id=state["workflow_id"]
        )
        
        return state
    
    def _route_from_start(self, state: WorkflowState) -> str:
        """Route workflow from start based on configuration."""
        if state.get("errors"):
            return "end"
        
        # Check if persona research is needed
        request = state["original_request"]
        if request.get("skip_research", False):
            return "content_strategy"
        
        return "persona_research"
    
    def _route_after_generation(self, state: WorkflowState) -> str:
        """Route after content generation - decide if QA is needed."""
        request = state["original_request"]
        
        # Check if QA should be skipped
        if request.get("skip_qa", False) or "quality_assurance" not in self.agents:
            return "synthesis"
        
        return "quality_assurance"
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of a running workflow."""
        if workflow_id not in self.active_workflows:
            return None
        
        workflow = self.active_workflows[workflow_id]
        return {
            "workflow_id": workflow_id,
            "status": workflow.workflow_status,
            "current_step": workflow.current_step,
            "progress": self._calculate_progress(workflow),
            "agents_completed": len(workflow.agent_responses),
            "errors": workflow.errors
        }
    
    def _calculate_progress(self, workflow: WorkflowState) -> float:
        """Calculate workflow progress percentage."""
        total_steps = 5  # start, research, strategy, generation, qa, synthesis
        
        step_progress = {
            "initialization": 0.1,
            "persona_research": 0.3,
            "content_strategy": 0.5, 
            "creative_generation": 0.7,
            "quality_assurance": 0.9,
            "synthesis": 0.95,
            "completed": 1.0
        }
        
        return step_progress.get(workflow.current_step, 0.0)
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow."""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.workflow_status = "cancelled"
            workflow.completed_at = datetime.utcnow()
            
            self.logger.workflow_complete(
                success=False,
                total_duration=0,
                results={"cancelled": True},
                workflow_id=workflow_id
            )
            
            return True
        
        return False
