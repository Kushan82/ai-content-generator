import asyncio
import uuid
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta

from app.agents.base_agent import BaseAgent, TaskRequest
from app.models.agent_schemas import AgentCapability, AgentStatus, AgentMessage
from app.services.llm_service import LLMService
from app.core.logging import get_workflow_logger, WorkflowLogger
from app.core.config import settings

class OrchestratorAgent(BaseAgent):
    """
    Master orchestrator agent that coordinates multi-agent workflows.
    
    This agent manages task delegation, agent communication, workflow state,
    and ensures optimal collaboration between specialized agents to achieve
    complex content generation objectives.
    """
    
    def __init__(self, llm_service: LLMService, available_agents: Dict[str, BaseAgent]):
        capabilities = [
            AgentCapability(
                name="workflow_orchestration",
                description="Orchestrate complex multi-agent workflows",
                proficiency_level=0.98,
                required_resources=["groq_api", "agent_registry"],
                average_execution_time=5.0
            ),
            AgentCapability(
                name="task_delegation",
                description="Intelligently delegate tasks to appropriate specialized agents",
                proficiency_level=0.96,
                required_resources=["agent_capabilities"],
                average_execution_time=3.0
            ),
            AgentCapability(
                name="agent_coordination",
                description="Coordinate communication and data flow between agents",
                proficiency_level=0.94,
                required_resources=["communication_protocols"],
                average_execution_time=2.0
            ),
            AgentCapability(
                name="quality_synthesis",
                description="Synthesize outputs from multiple agents into cohesive results",
                proficiency_level=0.92,
                required_resources=["groq_api"],
                average_execution_time=8.0
            )
        ]
        
        super().__init__(
            agent_id="orchestrator",
            name="Multi-Agent Workflow Orchestrator",
            role="Workflow Coordination & Agent Management",
            llm_service=llm_service,
            capabilities=capabilities
        )
        
        # Register available agents
        self.available_agents = available_agents
        self.workflow_logger = None
        
        # Orchestration patterns and strategies
        self.orchestration_patterns = {
            "sequential": self._execute_sequential_workflow,
            "parallel": self._execute_parallel_workflow,
            "conditional": self._execute_conditional_workflow,
            "iterative": self._execute_iterative_workflow
        }
        
        # Agent dependency mapping for workflow planning
        self.agent_dependencies = {
            "persona_researcher": [],  # No dependencies
            "content_strategist": ["persona_researcher"],  # Needs persona research
            "creative_generator": ["persona_researcher", "content_strategist"],  # Needs both
            "quality_assurance": ["creative_generator"]  # Needs generated content
        }
        
        # Workflow templates for different content generation scenarios
        self.workflow_templates = {
            "standard_content_generation": [
                "persona_researcher", "content_strategist", 
                "creative_generator", "quality_assurance"
            ],
            "rapid_content_generation": [
                "content_strategist", "creative_generator"
            ],
            "research_heavy_generation": [
                "persona_researcher", "content_strategist", 
                "creative_generator", "quality_assurance"
            ]
        }
        
        self.logger.info(
            "Orchestrator Agent initialized",
            available_agents=list(available_agents.keys()),
            workflow_patterns=list(self.orchestration_patterns.keys())
        )
    
    async def process_task(self, task: TaskRequest) -> Dict[str, Any]:
        """
        Orchestrate a complete multi-agent workflow for content generation.
        
        This method analyzes the task requirements, selects optimal workflow pattern,
        coordinates agent execution, and synthesizes final results.
        """
        workflow_id = str(uuid.uuid4())
        self.workflow_logger = get_workflow_logger(workflow_id)
        
        self.workflow_logger.workflow_start(
            workflow_type="content_generation",
            agents=list(self.available_agents.keys()),
            task_id=task.task_id
        )
        
        try:
            # Analyze task and determine optimal workflow
            workflow_plan = await self._analyze_and_plan_workflow(task)
            
            # Execute the planned workflow
            workflow_results = await self._execute_workflow(workflow_plan, task)
            
            # Synthesize final results from all agents
            final_result = await self._synthesize_results(workflow_results, task)
            
            # Calculate workflow performance metrics
            performance_metrics = self._calculate_workflow_metrics(workflow_results)
            
            result = {
                "final_content": final_result["synthesized_content"],
                "content_variations": final_result.get("variations", []),
                "workflow_summary": {
                    "workflow_id": workflow_id,
                    "pattern_used": workflow_plan["pattern"],
                    "agents_involved": workflow_plan["agents"],
                    "execution_order": workflow_plan["execution_order"],
                    "total_steps": len(workflow_plan["agents"]),
                    "success_rate": performance_metrics["success_rate"]
                },
                "agent_contributions": workflow_results,
                "quality_assessment": final_result.get("quality_report", {}),
                "orchestration_insights": {
                    "workflow_efficiency": performance_metrics["efficiency_score"],
                    "agent_coordination_score": performance_metrics["coordination_score"],
                    "optimization_opportunities": performance_metrics["optimization_suggestions"]
                },
                "performance_metrics": performance_metrics,
                "tokens_used": sum(result.get("tokens_used", 0) for result in workflow_results.values()),
                "api_calls": sum(result.get("api_calls", 0) for result in workflow_results.values())
            }
            
            self.workflow_logger.workflow_complete(
                success=True,
                total_duration=performance_metrics["total_duration"],
                results=result
            )
            
            return result
            
        except Exception as e:
            self.workflow_logger.workflow_error(e)
            self.logger.error(f"Workflow orchestration failed: {e}", workflow_id=workflow_id)
            raise
    
    async def _analyze_and_plan_workflow(self, task: TaskRequest) -> Dict[str, Any]:
        """
        Analyze task requirements and create optimal workflow execution plan.
        """
        task_data = task.input_data
        content_type = task_data.get("content_type", "generic")
        complexity_level = task_data.get("complexity_level", "standard")
        time_constraint = task_data.get("time_constraint", "normal")
        
        # Determine workflow pattern based on requirements
        if time_constraint == "urgent":
            pattern = "parallel"
            template = "rapid_content_generation"
        elif complexity_level == "high" or content_type in ["landing_page", "long_form"]:
            pattern = "sequential"
            template = "research_heavy_generation"
        else:
            pattern = "sequential"
            template = "standard_content_generation"
        
        # Get agent execution order from template
        agent_order = self.workflow_templates[template]
        
        # Filter available agents
        available_agent_ids = [
            agent_id for agent_id in agent_order 
            if agent_id in self.available_agents
        ]
        
        workflow_plan = {
            "pattern": pattern,
            "template": template,
            "agents": available_agent_ids,
            "execution_order": available_agent_ids,
            "estimated_duration": self._estimate_workflow_duration(available_agent_ids),
            "dependencies": self._map_agent_dependencies(available_agent_ids)
        }
        
        self.workflow_logger.workflow_step(
            step_name="workflow_planning",
            agent_id="orchestrator",
            status="completed",
            plan=workflow_plan
        )
        
        return workflow_plan
    
    async def _execute_workflow(self, workflow_plan: Dict[str, Any], task: TaskRequest) -> Dict[str, Dict[str, Any]]:
        """
        Execute the planned workflow using the specified orchestration pattern.
        """
        pattern = workflow_plan["pattern"]
        orchestration_func = self.orchestration_patterns[pattern]
        
        return await orchestration_func(workflow_plan, task)
    
    async def _execute_sequential_workflow(self, workflow_plan: Dict[str, Any], task: TaskRequest) -> Dict[str, Dict[str, Any]]:
        """
        Execute agents sequentially, passing outputs from one agent to the next.
        """
        workflow_results = {}
        accumulated_data = task.input_data.copy()
        
        for i, agent_id in enumerate(workflow_plan["agents"]):
            if agent_id not in self.available_agents:
                self.logger.warning(f"Agent {agent_id} not available, skipping")
                continue
            
            agent = self.available_agents[agent_id]
            step_name = f"step_{i+1}_{agent_id}"
            
            self.workflow_logger.workflow_step(
                step_name=step_name,
                agent_id=agent_id,
                status="starting"
            )
            
            # Create task for this agent with accumulated data
            agent_task = TaskRequest(
                task_type=self._get_agent_task_type(agent_id),
                input_data=accumulated_data,
                workflow_id=task.workflow_id,
                parent_task_id=task.task_id
            )
            
            try:
                # Execute agent task
                agent_result = await agent.execute_task(agent_task)
                workflow_results[agent_id] = agent_result.dict()
                
                # Accumulate data for next agent
                accumulated_data.update(self._extract_relevant_data(agent_result, agent_id))
                
                self.workflow_logger.workflow_step(
                    step_name=step_name,
                    agent_id=agent_id,
                    status="completed",
                    confidence=agent_result.confidence_score
                )
                
                # Log agent communication
                self.workflow_logger.agent_communication(
                    sender=agent_id,
                    receiver="orchestrator",
                    message_type="task_result"
                )
                
            except Exception as e:
                self.workflow_logger.workflow_step(
                    step_name=step_name,
                    agent_id=agent_id,
                    status="failed",
                    error=str(e)
                )
                
                # Attempt workflow recovery
                if await self._attempt_workflow_recovery(agent_id, e, accumulated_data):
                    continue
                else:
                    raise Exception(f"Workflow failed at agent {agent_id}: {e}")
        
        return workflow_results
    
    async def _execute_parallel_workflow(self, workflow_plan: Dict[str, Any], task: TaskRequest) -> Dict[str, Dict[str, Any]]:
        """
        Execute multiple agents in parallel for faster processing.
        """
        # Create tasks for all agents
        agent_tasks = []
        for agent_id in workflow_plan["agents"]:
            if agent_id in self.available_agents:
                agent = self.available_agents[agent_id]
                agent_task = TaskRequest(
                    task_type=self._get_agent_task_type(agent_id),
                    input_data=task.input_data,
                    workflow_id=task.workflow_id,
                    parent_task_id=task.task_id
                )
                agent_tasks.append((agent_id, agent.execute_task(agent_task)))
        
        # Execute all tasks concurrently
        results = await asyncio.gather(
            *[task_coroutine for _, task_coroutine in agent_tasks],
            return_exceptions=True
        )
        
        # Process results
        workflow_results = {}
        for i, (agent_id, result) in enumerate(zip([aid for aid, _ in agent_tasks], results)):
            if isinstance(result, Exception):
                self.logger.error(f"Agent {agent_id} failed: {result}")
                workflow_results[agent_id] = {"error": str(result), "success": False}
            else:
                workflow_results[agent_id] = result.dict()
        
        return workflow_results
    
    async def _execute_conditional_workflow(self, workflow_plan: Dict[str, Any], task: TaskRequest) -> Dict[str, Dict[str, Any]]:
        """
        Execute workflow with conditional branching based on intermediate results.
        """
        # This would implement conditional logic based on agent outputs
        # For now, fall back to sequential execution
        return await self._execute_sequential_workflow(workflow_plan, task)
    
    async def _execute_iterative_workflow(self, workflow_plan: Dict[str, Any], task: TaskRequest) -> Dict[str, Dict[str, Any]]:
        """
        Execute workflow with iterative refinement loops.
        """
        max_iterations = 2
        workflow_results = {}
        
        for iteration in range(max_iterations):
            iteration_results = await self._execute_sequential_workflow(workflow_plan, task)
            workflow_results[f"iteration_{iteration+1}"] = iteration_results
            
            # Check if quality threshold is met
            if self._check_quality_threshold(iteration_results):
                break
        
        # Return best iteration results
        return workflow_results.get("iteration_2", workflow_results["iteration_1"])
    
    async def _synthesize_results(self, workflow_results: Dict[str, Dict[str, Any]], task: TaskRequest) -> Dict[str, Any]:
        """
        Synthesize outputs from multiple agents into a cohesive final result.
        """
        synthesis_prompt = f"""
        As the orchestrator of a multi-agent content generation system, synthesize the outputs 
        from specialized agents into the final, optimized content:
        
        ORIGINAL TASK: {task.input_data.get('topic', 'Content Generation')}
        CONTENT TYPE: {task.input_data.get('content_type', 'Generic')}
        
        AGENT OUTPUTS:
        {self._format_agent_outputs_for_synthesis(workflow_results)}
        
        Create the final synthesized result that:
        1. Uses the highest-quality generated content as the base
        2. Incorporates insights from persona research
        3. Applies strategic recommendations
        4. Addresses quality assurance feedback
        5. Maintains coherence and consistency
        
        Provide response in JSON format:
        {{
            "synthesized_content": "final optimized content",
            "synthesis_rationale": "explanation of how outputs were combined",
            "quality_improvements": ["list of improvements made"],
            "confidence_score": 0.0-1.0
        }}
        """
        
        try:
            synthesis_response = await self.llm_service.generate_content(
                prompt=synthesis_prompt,
                model=settings.GROQ_MODEL_SMART,
                temperature=0.3,
                max_tokens=1000
            )
            
            import json
            synthesis_result = json.loads(synthesis_response.content)
            
            return synthesis_result
            
        except Exception as e:
            self.logger.error(f"Result synthesis failed: {e}")
            
            # Fallback: return best available content
            return self._fallback_synthesis(workflow_results)
    
    def _fallback_synthesis(self, workflow_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Fallback synthesis when LLM synthesis fails."""
        
        # Find the best content from available results
        best_content = ""
        best_score = 0.0
        
        for agent_id, result in workflow_results.items():
            content = result.get("content", {})
            
            if agent_id == "creative_generator" and content.get("generated_content"):
                best_content = content["generated_content"]
                best_score = result.get("confidence_score", 0.8)
            elif agent_id == "quality_assurance" and content.get("optimized_content"):
                best_content = content["optimized_content"]
                best_score = result.get("confidence_score", 0.9)
        
        return {
            "synthesized_content": best_content,
            "synthesis_rationale": "Fallback synthesis - used highest quality available content",
            "quality_improvements": ["Applied best available content"],
            "confidence_score": best_score
        }
    
    def _format_agent_outputs_for_synthesis(self, workflow_results: Dict[str, Dict[str, Any]]) -> str:
        """Format agent outputs for synthesis prompt."""
        formatted_outputs = []
        
        for agent_id, result in workflow_results.items():
            content = result.get("content", {})
            confidence = result.get("confidence_score", 0.0)
            
            formatted_output = f"""
            AGENT: {agent_id.upper()}
            Confidence: {confidence:.2f}
            Output: {str(content)[:500]}...
            """
            formatted_outputs.append(formatted_output)
        
        return "\n".join(formatted_outputs)
    
    def _get_agent_task_type(self, agent_id: str) -> str:
        """Get appropriate task type for each agent."""
        task_type_mapping = {
            "persona_researcher": "persona_research",
            "content_strategist": "strategy_development", 
            "creative_generator": "content_generation",
            "quality_assurance": "quality_assessment"
        }
        return task_type_mapping.get(agent_id, "generic_task")
    
    def _extract_relevant_data(self, agent_result, agent_id: str) -> Dict[str, Any]:
        """Extract relevant data from agent result for next agent."""
        content = agent_result.content
        
        if agent_id == "persona_researcher":
            return {"persona_research": content}
        elif agent_id == "content_strategist":
            return {"content_strategy": content}
        elif agent_id == "creative_generator":
            return {"generated_content": content.get("generated_content", "")}
        elif agent_id == "quality_assurance":
            return {"quality_report": content}
        
        return {}
    
    def _estimate_workflow_duration(self, agent_ids: List[str]) -> float:
        """Estimate total workflow duration based on agent capabilities."""
        total_duration = 0.0
        
        for agent_id in agent_ids:
            if agent_id in self.available_agents:
                agent = self.available_agents[agent_id]
                # Use average execution time from agent capabilities
                for capability in agent.capabilities:
                    total_duration += capability.average_execution_time
                    break  # Use first capability as estimate
        
        return total_duration
    
    def _map_agent_dependencies(self, agent_ids: List[str]) -> Dict[str, List[str]]:
        """Map dependencies between agents in the workflow."""
        return {
            agent_id: self.agent_dependencies.get(agent_id, [])
            for agent_id in agent_ids
        }
    
    async def _attempt_workflow_recovery(self, failed_agent_id: str, error: Exception, data: Dict[str, Any]) -> bool:
        """Attempt to recover from agent failures."""
        self.logger.info(f"Attempting recovery from {failed_agent_id} failure")
        
        # Simple recovery: skip non-critical agents
        non_critical_agents = ["quality_assurance"]
        
        if failed_agent_id in non_critical_agents:
            self.logger.info(f"Skipping non-critical agent {failed_agent_id}")
            return True
        
        return False
    
    def _check_quality_threshold(self, results: Dict[str, Dict[str, Any]]) -> bool:
        """Check if results meet quality threshold for iterative workflows."""
        if "quality_assurance" in results:
            qa_result = results["quality_assurance"]
            overall_score = qa_result.get("content", {}).get("quality_assessment", {}).get("overall_score", 0.0)
            return overall_score >= 0.85
        
        return False
    
    def _calculate_workflow_metrics(self, workflow_results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive workflow performance metrics."""
        
        successful_agents = sum(1 for result in workflow_results.values() 
                              if result.get("status") == "completed")
        total_agents = len(workflow_results)
        
        success_rate = successful_agents / max(total_agents, 1)
        
        # Calculate average confidence
        confidence_scores = [result.get("confidence_score", 0.0) 
                           for result in workflow_results.values()]
        avg_confidence = sum(confidence_scores) / max(len(confidence_scores), 1)
        
        # Calculate efficiency score based on success rate and confidence
        efficiency_score = (success_rate * 0.6 + avg_confidence * 0.4)
        
        # Coordination score based on successful handoffs
        coordination_score = min(success_rate + 0.1, 1.0)
        
        return {
            "total_agents": total_agents,
            "successful_agents": successful_agents,
            "success_rate": success_rate,
            "average_confidence": avg_confidence,
            "efficiency_score": efficiency_score,
            "coordination_score": coordination_score,
            "total_duration": sum(result.get("processing_time", 0) for result in workflow_results.values()),
            "optimization_suggestions": self._generate_workflow_optimization_suggestions(workflow_results)
        }
    
    def _generate_workflow_optimization_suggestions(self, workflow_results: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate suggestions for workflow optimization."""
        suggestions = []
        
        # Analyze failure patterns
        failed_agents = [agent_id for agent_id, result in workflow_results.items() 
                        if result.get("status") == "error"]
        
        if failed_agents:
            suggestions.append(f"Investigate failures in: {', '.join(failed_agents)}")
        
        # Analyze confidence patterns
        low_confidence_agents = [agent_id for agent_id, result in workflow_results.items()
                               if result.get("confidence_score", 1.0) < 0.7]
        
        if low_confidence_agents:
            suggestions.append(f"Improve confidence in: {', '.join(low_confidence_agents)}")
        
        # Performance suggestions
        slow_agents = [agent_id for agent_id, result in workflow_results.items()
                      if result.get("processing_time", 0) > 30]
        
        if slow_agents:
            suggestions.append(f"Optimize performance for: {', '.join(slow_agents)}")
        
        return suggestions[:5]  # Return top 5 suggestions
