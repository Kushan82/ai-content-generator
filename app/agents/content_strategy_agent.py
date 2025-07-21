import json
from typing import Dict, Any, List
from datetime import datetime

from app.agents.base_agent import BaseAgent, TaskRequest
from app.models.agent_schemas import AgentCapability, AgentStatus
from app.services.llm_service import LLMService, LLMRequest
from app.core.config import settings

class ContentStrategyAgent(BaseAgent):
    """
    Strategic content planning agent that develops comprehensive messaging strategies.
    
    This agent takes persona research and transforms it into actionable content strategies,
    defining messaging frameworks, persuasion techniques, and structural blueprints
    for optimal content performance.
    """
    
    def __init__(self, llm_service: LLMService):
        capabilities = [
            AgentCapability(
                name="messaging_strategy",
                description="Develop core messaging and value proposition frameworks",
                proficiency_level=0.92,
                required_resources=["groq_api", "persona_data"],
                average_execution_time=18.0
            ),
            AgentCapability(
                name="content_planning",
                description="Create detailed content structure and format strategies",
                proficiency_level=0.90,
                required_resources=["groq_api", "persona_data"],
                average_execution_time=15.0
            ),
            AgentCapability(
                name="persuasion_optimization",
                description="Apply psychological principles for maximum persuasive impact",
                proficiency_level=0.88,
                required_resources=["groq_api"],
                average_execution_time=12.0
            ),
            AgentCapability(
                name="competitive_positioning",
                description="Position content strategically against competitors",
                proficiency_level=0.85,
                required_resources=["groq_api", "market_data"],
                average_execution_time=20.0
            )
        ]
        
        super().__init__(
            agent_id="content_strategist",
            name="Content Strategy Planner",
            role="Strategic Content Planning & Messaging Architecture",
            llm_service=llm_service,
            capabilities=capabilities
        )
        
        # Strategy frameworks and methodologies
        self.strategy_frameworks = {
            "messaging": self._get_messaging_framework(),
            "structure": self._get_structure_framework(),
            "persuasion": self._get_persuasion_framework(),
            "positioning": self._get_positioning_framework()
        }
        
        # Content type specific strategies
        self.content_strategies = {
            "ad": self._get_ad_strategy_template(),
            "landing_page": self._get_landing_strategy_template(),
            "blog_intro": self._get_blog_strategy_template(),
            "email": self._get_email_strategy_template(),
            "social_media": self._get_social_strategy_template()
        }
        
        self.logger.info("Content Strategy Agent initialized with comprehensive strategy frameworks")
    
    async def process_task(self, task: TaskRequest) -> Dict[str, Any]:
        """
        Develop comprehensive content strategy based on persona research and content requirements.
        
        This method analyzes persona data, content type, and business objectives to create
        a detailed strategic blueprint for content creation.
        """
        self.logger.info(
            "Starting content strategy development",
            task_id=task.task_id,
            content_type=task.input_data.get("content_type"),
            has_persona_data=bool(task.input_data.get("persona_research"))
        )
        
        # Extract input data
        persona_research = task.input_data.get("persona_research", {})
        content_type = task.input_data.get("content_type")
        topic = task.input_data.get("topic")
        additional_context = task.input_data.get("additional_context", "")
        business_objectives = task.input_data.get("business_objectives", [])
        
        if not persona_research:
            raise ValueError("persona_research is required for strategy development")
        if not content_type or not topic:
            raise ValueError("content_type and topic are required for strategy development")
        
        # Build comprehensive strategy prompt
        strategy_prompt = self._build_strategy_prompt(
            persona_research, content_type, topic, additional_context, business_objectives
        )
        
        # Execute strategy development using smart model
        llm_request = LLMRequest(
            prompt=strategy_prompt,
            model=settings.GROQ_MODEL_SMART,
            temperature=0.4,  # Balanced creativity for strategic thinking
            max_tokens=1800,
            system_prompt=self._get_strategist_system_prompt()
        )
        
        try:
            response = await self.llm_service.generate_content(
                prompt=strategy_prompt,
                model=settings.GROQ_MODEL_SMART,
                temperature=0.4,
                max_tokens=1800,
                task_type="strategy"
            )
            
            # Parse and enhance strategy results
            strategy_data = await self._parse_strategy_results(response.content, content_type)
            
            # Add strategic enhancements and optimizations
            enhanced_strategy = await self._enhance_strategy_data(
                strategy_data, persona_research, content_type, topic
            )
            
            # Store strategy in memory for cross-reference
            strategy_key = f"content_strategy_{content_type}_{hash(topic)}"
            self.add_memory(strategy_key, enhanced_strategy, ttl=1800)  # 30 minutes
            
            result = {
                "content_strategy": enhanced_strategy,
                "strategic_confidence": self._calculate_strategy_confidence(enhanced_strategy),
                "strategic_approach": self._identify_primary_approach(enhanced_strategy),
                "optimization_score": self._calculate_optimization_score(enhanced_strategy),
                "implementation_priority": self._assess_implementation_priority(enhanced_strategy),
                "expected_performance": self._predict_content_performance(enhanced_strategy, persona_research),
                "tokens_used": response.tokens_used,
                "api_calls": 1,
                "strategy_metadata": {
                    "content_type": content_type,
                    "topic": topic,
                    "persona_alignment_score": self._calculate_persona_alignment(enhanced_strategy, persona_research),
                    "competitive_advantage": self._assess_competitive_advantage(enhanced_strategy),
                    "timestamp": datetime.utcnow().isoformat()
                }
            }
            
            self.logger.info(
                "Content strategy development completed",
                task_id=task.task_id,
                confidence=result["strategic_confidence"],
                approach=result["strategic_approach"],
                performance_prediction=result["expected_performance"].get("overall_score", 0)
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Content strategy development failed",
                task_id=task.task_id,
                error=str(e),
                content_type=content_type
            )
            raise
    
    def _build_strategy_prompt(
        self, 
        persona_research: Dict[str, Any], 
        content_type: str, 
        topic: str,
        additional_context: str,
        business_objectives: List[str]
    ) -> str:
        """
        Build comprehensive strategy development prompt incorporating all available data.
        """
        # Extract key persona insights for strategy focus
        key_pain_points = persona_research.get("pain_points", [])[:3]
        key_goals = persona_research.get("goals", [])[:3] 
        communication_prefs = persona_research.get("communication_preferences", {})
        psychographics = persona_research.get("psychographics", {})
        
        strategy_prompt = f"""
        As a senior content strategist, develop a comprehensive strategy for creating {content_type} content about "{topic}".
        
        PERSONA RESEARCH INSIGHTS:
        Key Pain Points: {', '.join(key_pain_points)}
        Primary Goals: {', '.join(key_goals)}
        Communication Style: {communication_prefs}
        Values & Interests: {psychographics}
        
        {"Additional Context: " + additional_context if additional_context else ""}
        {"Business Objectives: " + ', '.join(business_objectives) if business_objectives else ""}
        
        STRATEGIC FRAMEWORK:
        {self.strategy_frameworks["messaging"]}
        {self.strategy_frameworks["structure"]}
        {self.strategy_frameworks["persuasion"]}
        {self.strategy_frameworks["positioning"]}
        
        CONTENT TYPE SPECIFICATIONS:
        {self.content_strategies.get(content_type, self._get_generic_strategy_template())}
        
        Provide a comprehensive strategy formatted as JSON with these sections:
        
        {{
            "core_messaging": {{
                "primary_value_proposition": "main benefit statement",
                "supporting_messages": ["key supporting points"],
                "unique_selling_proposition": "differentiation factor",
                "emotional_hooks": ["emotional triggers to use"]
            }},
            "content_structure": {{
                "opening_strategy": "how to start for maximum impact",
                "body_framework": "main content organization approach", 
                "closing_strategy": "how to end with strong CTA",
                "optimal_length": "recommended word/character count",
                "key_sections": ["required content sections"]
            }},
            "persuasion_techniques": {{
                "primary_technique": "main psychological principle to apply",
                "supporting_techniques": ["additional persuasion methods"],
                "social_proof_strategy": "how to incorporate credibility",
                "urgency_approach": "how to create appropriate urgency",
                "trust_building": "methods to establish credibility"
            }},
            "messaging_optimization": {{
                "tone_specification": "exact tone to use",
                "language_level": "complexity and vocabulary level",
                "cultural_considerations": ["cultural factors to consider"],
                "personalization_points": ["where to customize for persona"],
                "power_words": ["high-impact words to include"]
            }},
            "call_to_action": {{
                "primary_cta": "main action to request",
                "cta_positioning": "where to place the CTA",
                "cta_language": "exact wording to use",
                "urgency_elements": ["urgency factors to include"],
                "risk_mitigation": ["ways to reduce perceived risk"]
            }},
            "competitive_strategy": {{
                "differentiation_points": ["how to stand out from competitors"],
                "competitive_advantages": ["unique strengths to emphasize"],
                "market_positioning": "where to position in the market",
                "objection_handling": ["anticipated objections and responses"]
            }},
            "optimization_recommendations": {{
                "a_b_test_suggestions": ["elements to test"],
                "performance_indicators": ["metrics to track"],
                "improvement_opportunities": ["areas for enhancement"],
                "risk_factors": ["potential issues to monitor"]
            }}
        }}
        
        Ensure all recommendations are specific, actionable, and directly tied to the persona research insights.
        """
        
        return strategy_prompt
    
    def _get_strategist_system_prompt(self) -> str:
        """System prompt defining the strategist's expertise and approach"""
        return """
        You are a world-class content strategist with 20+ years of experience in developing 
        high-converting marketing content across all industries and formats. Your expertise includes:
        
        - Advanced behavioral psychology and persuasion principles
        - Data-driven content optimization and A/B testing
        - Cross-channel content strategy and messaging architecture
        - Competitive analysis and market positioning
        - Conversion rate optimization and funnel strategy
        - Brand voice development and messaging consistency
        
        Your strategic approach is always:
        - Deeply rooted in audience psychology and persona insights
        - Focused on measurable business outcomes
        - Competitive and differentiated in the marketplace
        - Optimized for the specific content format and channel
        - Scalable and consistent across campaigns
        - Testing-oriented with clear optimization pathways
        
        You create strategies that consistently outperform benchmarks through psychological insight,
        strategic positioning, and tactical excellence.
        """
    
    async def _parse_strategy_results(self, raw_response: str, content_type: str) -> Dict[str, Any]:
        """
        Parse and structure the raw strategy response into organized strategic data.
        """
        try:
            strategy_data = json.loads(raw_response)
            
            # Validate required strategic sections
            required_sections = [
                "core_messaging", "content_structure", "persuasion_techniques",
                "messaging_optimization", "call_to_action"
            ]
            
            for section in required_sections:
                if section not in strategy_data:
                    strategy_data[section] = {}
            
            return strategy_data
            
        except json.JSONDecodeError:
            self.logger.warning(
                "Failed to parse JSON strategy response, using text analysis fallback",
                content_type=content_type
            )
            
            return await self._extract_strategy_from_text(raw_response, content_type)
    
    async def _extract_strategy_from_text(self, text: str, content_type: str) -> Dict[str, Any]:
        """
        Extract strategic elements from unstructured text using secondary LLM parsing.
        """
        parsing_prompt = f"""
        Extract strategic elements from this content strategy analysis and format as JSON:
        
        {text}
        
        Create a JSON object with these exact sections:
        - core_messaging: object with value proposition and key messages
        - content_structure: object with opening, body, closing strategies  
        - persuasion_techniques: object with psychological principles to apply
        - messaging_optimization: object with tone and language specifications
        - call_to_action: object with CTA strategy and placement
        - competitive_strategy: object with differentiation and positioning
        
        Focus on actionable, specific strategic recommendations.
        """
        
        try:
            parsing_response = await self.llm_service.generate_content(
                prompt=parsing_prompt,
                model=settings.GROQ_MODEL_FAST,
                temperature=0.1,
                max_tokens=1000
            )
            
            return json.loads(parsing_response.content)
            
        except Exception as e:
            self.logger.error("Strategy parsing fallback failed", error=str(e))
            
            # Final fallback with basic structure
            return {
                "core_messaging": {
                    "primary_value_proposition": "Value-focused approach",
                    "supporting_messages": ["Benefit 1", "Benefit 2"],
                    "unique_selling_proposition": "Competitive advantage"
                },
                "content_structure": {
                    "opening_strategy": "Problem-focused opening",
                    "body_framework": "Benefit-driven structure",
                    "closing_strategy": "Strong call-to-action"
                },
                "persuasion_techniques": {
                    "primary_technique": "Social proof",
                    "supporting_techniques": ["Authority", "Scarcity"]
                },
                "raw_analysis": text,
                "parsing_error": str(e)
            }
    
    async def _enhance_strategy_data(
        self, 
        strategy_data: Dict[str, Any], 
        persona_research: Dict[str, Any],
        content_type: str,
        topic: str
    ) -> Dict[str, Any]:
        """
        Enhance base strategy with additional strategic insights and optimizations.
        """
        enhanced_strategy = strategy_data.copy()
        
        # Add strategic calculations and insights
        enhanced_strategy["strategic_insights"] = {
            "persona_alignment_factors": self._identify_alignment_factors(strategy_data, persona_research),
            "psychological_triggers": self._identify_psychological_triggers(strategy_data, persona_research),
            "content_optimization_points": self._identify_optimization_points(strategy_data, content_type),
            "competitive_differentiators": self._identify_differentiators(strategy_data, topic)
        }
        
        # Add implementation guidance
        enhanced_strategy["implementation_guide"] = {
            "priority_order": self._determine_implementation_priority(strategy_data),
            "resource_requirements": self._assess_resource_requirements(strategy_data),
            "timeline_recommendations": self._suggest_implementation_timeline(strategy_data),
            "success_metrics": self._define_success_metrics(strategy_data, content_type)
        }
        
        # Add risk assessment
        enhanced_strategy["risk_assessment"] = {
            "potential_issues": self._identify_potential_issues(strategy_data, persona_research),
            "mitigation_strategies": self._suggest_risk_mitigation(strategy_data),
            "contingency_plans": self._develop_contingency_plans(strategy_data)
        }
        
        return enhanced_strategy
    
    def _calculate_strategy_confidence(self, strategy_data: Dict[str, Any]) -> float:
        """Calculate confidence score for the strategic recommendations"""
        completeness = self._assess_strategy_completeness(strategy_data)
        specificity = self._assess_strategy_specificity(strategy_data)
        coherence = self._assess_strategy_coherence(strategy_data)
        
        confidence = (completeness * 0.4 + specificity * 0.35 + coherence * 0.25)
        return min(max(confidence, 0.2), 0.95)
    
    def _identify_primary_approach(self, strategy_data: Dict[str, Any]) -> str:
        """Identify the primary strategic approach being recommended"""
        persuasion_techniques = strategy_data.get("persuasion_techniques", {})
        primary_technique = persuasion_techniques.get("primary_technique", "").lower()
        
        if "emotion" in primary_technique or "empathy" in primary_technique:
            return "emotion-driven"
        elif "logic" in primary_technique or "rational" in primary_technique:
            return "logic-driven"
        elif "social" in primary_technique or "proof" in primary_technique:
            return "social-proof-focused"
        elif "authority" in primary_technique or "expert" in primary_technique:
            return "authority-based"
        else:
            return "balanced-approach"
    
    def _calculate_optimization_score(self, strategy_data: Dict[str, Any]) -> float:
        """Calculate how well-optimized the strategy is for performance"""
        optimization_factors = []
        
        # Check for CTA optimization
        cta_section = strategy_data.get("call_to_action", {})
        if cta_section.get("primary_cta") and cta_section.get("cta_positioning"):
            optimization_factors.append(1.0)
        else:
            optimization_factors.append(0.5)
        
        # Check for persuasion optimization
        persuasion_section = strategy_data.get("persuasion_techniques", {})
        if len(persuasion_section.get("supporting_techniques", [])) >= 2:
            optimization_factors.append(1.0)
        else:
            optimization_factors.append(0.6)
        
        # Check for messaging optimization
        messaging_section = strategy_data.get("messaging_optimization", {})
        if messaging_section.get("tone_specification") and messaging_section.get("power_words"):
            optimization_factors.append(1.0)
        else:
            optimization_factors.append(0.7)
        
        return sum(optimization_factors) / len(optimization_factors) if optimization_factors else 0.7
    
    def _assess_implementation_priority(self, strategy_data: Dict[str, Any]) -> str:
        """Assess implementation priority based on strategic complexity and impact"""
        complexity_indicators = [
            len(strategy_data.get("core_messaging", {}).get("supporting_messages", [])),
            len(strategy_data.get("persuasion_techniques", {}).get("supporting_techniques", [])),
            len(strategy_data.get("competitive_strategy", {}).get("differentiation_points", []))
        ]
        
        avg_complexity = sum(complexity_indicators) / len(complexity_indicators)
        
        if avg_complexity >= 4:
            return "high"
        elif avg_complexity >= 2:
            return "medium"
        else:
            return "low"
    
    def _predict_content_performance(
        self, 
        strategy_data: Dict[str, Any], 
        persona_research: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict expected content performance based on strategy and persona alignment"""
        
        # Calculate alignment scores
        messaging_alignment = self._calculate_messaging_alignment(strategy_data, persona_research)
        persuasion_alignment = self._calculate_persuasion_alignment(strategy_data, persona_research)
        tone_alignment = self._calculate_tone_alignment(strategy_data, persona_research)
        
        # Overall performance prediction
        overall_score = (messaging_alignment * 0.4 + persuasion_alignment * 0.35 + tone_alignment * 0.25)
        
        return {
            "overall_score": overall_score,
            "messaging_alignment": messaging_alignment,
            "persuasion_effectiveness": persuasion_alignment,
            "tone_appropriateness": tone_alignment,
            "predicted_engagement": "high" if overall_score >= 0.8 else "medium" if overall_score >= 0.6 else "low",
            "optimization_potential": max(0, (1.0 - overall_score) * 100),
            "confidence_level": "high" if overall_score >= 0.75 else "moderate"
        }
    
    # Strategy Framework Templates
    def _get_messaging_framework(self) -> str:
        return """
        CORE MESSAGING STRATEGY:
        - Primary value proposition that resonates with persona's top goal
        - Secondary messages addressing key pain points
        - Unique selling proposition leveraging competitive advantages
        - Emotional connection points based on psychographic profile
        """
    
    def _get_structure_framework(self) -> str:
        return """
        CONTENT STRUCTURE STRATEGY:
        - Opening hook that immediately addresses persona's primary concern
        - Logical flow that matches persona's decision-making process
        - Information hierarchy prioritizing most relevant benefits
        - Closing strategy that removes final objections and motivates action
        """
    
    def _get_persuasion_framework(self) -> str:
        return """
        PERSUASION OPTIMIZATION STRATEGY:
        - Primary psychological principle matching persona's motivation drivers
        - Social proof strategy relevant to persona's reference groups
        - Authority indicators that resonate with persona's trust factors
        - Urgency and scarcity tactics appropriate to persona's decision timeline
        """
    
    def _get_positioning_framework(self) -> str:
        return """
        COMPETITIVE POSITIONING STRATEGY:
        - Market differentiation based on unique persona insights
        - Competitive advantages most relevant to persona's priorities
        - Objection handling for common persona concerns
        - Strategic positioning to avoid direct competitor confrontation
        """
    
    # Content Type Strategy Templates
    def _get_ad_strategy_template(self) -> str:
        return """
        ADVERTISEMENT STRATEGY SPECIFICATIONS:
        - Attention-grabbing headline within first 3 seconds
        - Single primary message with emotional hook
        - Visual and textual elements working in harmony
        - Strong, clear call-to-action with urgency element
        - Length: 25-50 words for maximum impact
        """
    
    def _get_landing_strategy_template(self) -> str:
        return """
        LANDING PAGE STRATEGY SPECIFICATIONS:
        - Compelling headline matching ad/email promise
        - Subheadline clarifying value proposition
        - 3-5 key benefits with social proof integration
        - Single, prominent call-to-action above fold
        - Length: 200-500 words optimized for conversion
        """
    
    def _get_blog_strategy_template(self) -> str:
        return """
        BLOG INTRODUCTION STRATEGY SPECIFICATIONS:
        - Hook addressing persona's immediate interest/problem
        - Promise of specific, actionable value
        - Credibility establishment through expertise demonstration
        - Smooth transition to main content
        - Length: 100-200 words setting up full article
        """
    
    def _get_email_strategy_template(self) -> str:
        return """
        EMAIL STRATEGY SPECIFICATIONS:
        - Subject line creating curiosity without being clickbait
        - Personal greeting acknowledging persona's situation
        - Primary message with single call-to-action
        - Mobile-optimized formatting and length
        - Length: 100-300 words for optimal engagement
        """
    
    def _get_social_strategy_template(self) -> str:
        return """
        SOCIAL MEDIA STRATEGY SPECIFICATIONS:
        - Platform-specific format optimization
        - Visual-first approach with supporting text
        - Hashtag strategy for persona's interest communities
        - Engagement-focused call-to-action
        - Length: Platform-optimized (Twitter: 280, LinkedIn: 200, etc.)
        """
    
    def _get_generic_strategy_template(self) -> str:
        return """
        GENERAL CONTENT STRATEGY SPECIFICATIONS:
        - Clear value proposition aligned with persona goals
        - Structured approach addressing pain points
        - Appropriate tone and complexity level
        - Strong call-to-action driving desired behavior
        - Optimized length for format and audience attention span
        """
    
    # Helper methods for strategic analysis
    def _calculate_persona_alignment(self, strategy: Dict[str, Any], persona: Dict[str, Any]) -> float:
        """Calculate how well strategy aligns with persona characteristics"""
        # Simplified alignment calculation - could be enhanced with more sophisticated analysis
        alignment_score = 0.8  # Base alignment
        
        # Check tone alignment
        strategy_tone = strategy.get("messaging_optimization", {}).get("tone_specification", "").lower()
        persona_tone = str(persona.get("communication_preferences", {}).get("tone", "")).lower()
        
        if strategy_tone and persona_tone and strategy_tone in persona_tone:
            alignment_score += 0.1
        
        # Check pain point addressing
        core_messaging = strategy.get("core_messaging", {})
        persona_pain_points = persona.get("pain_points", [])
        
        if core_messaging and persona_pain_points:
            messaging_str = str(core_messaging).lower()
            pain_points_addressed = sum(1 for pain in persona_pain_points 
                                     if any(word in messaging_str for word in str(pain).lower().split()))
            if pain_points_addressed > 0:
                alignment_score += min(0.1, pain_points_addressed / len(persona_pain_points) * 0.2)
        
        return min(alignment_score, 1.0)
    
    def _assess_competitive_advantage(self, strategy: Dict[str, Any]) -> str:
        """Assess the competitive advantage provided by the strategy"""
        competitive_strategy = strategy.get("competitive_strategy", {})
        
        differentiation_points = competitive_strategy.get("differentiation_points", [])
        competitive_advantages = competitive_strategy.get("competitive_advantages", [])
        
        total_advantages = len(differentiation_points) + len(competitive_advantages)
        
        if total_advantages >= 4:
            return "strong"
        elif total_advantages >= 2:
            return "moderate"
        else:
            return "limited"
    
    # Additional helper methods would continue here for completeness...
    def _assess_strategy_completeness(self, strategy_data: Dict[str, Any]) -> float:
        """Assess completeness of strategic recommendations"""
        required_sections = [
            "core_messaging", "content_structure", "persuasion_techniques",
            "messaging_optimization", "call_to_action"
        ]
        
        present_sections = sum(1 for section in required_sections 
                             if section in strategy_data and strategy_data[section])
        
        return present_sections / len(required_sections)
    
    def _assess_strategy_specificity(self, strategy_data: Dict[str, Any]) -> float:
        """Assess how specific and actionable the strategy is"""
        specificity_score = 0.0
        
        # Check for specific recommendations vs generic advice
        messaging = strategy_data.get("core_messaging", {})
        if messaging.get("primary_value_proposition") and len(messaging.get("primary_value_proposition", "")) > 20:
            specificity_score += 0.3
        
        persuasion = strategy_data.get("persuasion_techniques", {})
        if len(persuasion.get("supporting_techniques", [])) >= 2:
            specificity_score += 0.3
        
        cta = strategy_data.get("call_to_action", {})
        if cta.get("cta_language") and cta.get("cta_positioning"):
            specificity_score += 0.4
        
        return specificity_score
    
    def _assess_strategy_coherence(self, strategy_data: Dict[str, Any]) -> float:
        """Assess internal coherence and consistency of the strategy"""
        # Simplified coherence check - could be enhanced
        coherence_score = 0.85  # Base coherence assumption
        
        # Check for consistency between messaging and persuasion techniques
        messaging_tone = strategy_data.get("messaging_optimization", {}).get("tone_specification", "").lower()
        primary_technique = strategy_data.get("persuasion_techniques", {}).get("primary_technique", "").lower()
        
        # Simple consistency check
        if "emotional" in primary_technique and "professional" in messaging_tone:
            coherence_score -= 0.1  # Slight inconsistency
        
        return coherence_score
    
    # More helper methods for enhancement calculations...
    def _identify_alignment_factors(self, strategy: Dict[str, Any], persona: Dict[str, Any]) -> List[str]:
        """Identify key factors where strategy aligns with persona"""
        factors = []
        
        # Check goal alignment
        persona_goals = persona.get("goals", [])
        core_messaging = str(strategy.get("core_messaging", {})).lower()
        
        for goal in persona_goals[:3]:  # Check top 3 goals
            if any(word in core_messaging for word in str(goal).lower().split()):
                factors.append(f"Addresses persona goal: {goal}")
        
        # Check communication preference alignment
        persona_comm = persona.get("communication_preferences", {})
        strategy_optimization = strategy.get("messaging_optimization", {})
        
        if persona_comm.get("tone") and strategy_optimization.get("tone_specification"):
            factors.append("Tone alignment with persona preferences")
        
        return factors
    
    def _identify_psychological_triggers(self, strategy: Dict[str, Any], persona: Dict[str, Any]) -> List[str]:
        """Identify psychological triggers that will be most effective"""
        triggers = []
        
        # Based on persona psychographics and strategy techniques
        psychographics = persona.get("psychographics", {})
        persuasion_techniques = strategy.get("persuasion_techniques", {})
        
        primary_technique = persuasion_techniques.get("primary_technique", "")
        triggers.append(f"Primary trigger: {primary_technique}")
        
        # Add supporting triggers
        supporting = persuasion_techniques.get("supporting_techniques", [])
        for technique in supporting[:2]:  # Top 2 supporting techniques
            triggers.append(f"Supporting trigger: {technique}")
        
        return triggers
    
    def _identify_optimization_points(self, strategy: Dict[str, Any], content_type: str) -> List[str]:
        """Identify key areas for content optimization"""
        optimization_points = []
        
        # Content-type specific optimizations
        if content_type == "ad":
            optimization_points.extend([
                "Headline A/B testing opportunities",
                "Visual-text balance optimization",
                "CTA button positioning"
            ])
        elif content_type == "landing_page":
            optimization_points.extend([
                "Above-fold content optimization",
                "Form field minimization",
                "Social proof placement"
            ])
        elif content_type == "email":
            optimization_points.extend([
                "Subject line optimization",
                "Mobile formatting",
                "Send time optimization"
            ])
        
        # General optimizations
        optimization_points.extend([
            "Persona-specific personalization",
            "Conversion funnel alignment",
            "Multi-variant testing setup"
        ])
        
        return optimization_points[:5]  # Return top 5 recommendations
    
    def _identify_differentiators(self, strategy: Dict[str, Any], topic: str) -> List[str]:
        """Identify key competitive differentiators"""
        competitive_strategy = strategy.get("competitive_strategy", {})
        
        differentiators = competitive_strategy.get("differentiation_points", [])
        advantages = competitive_strategy.get("competitive_advantages", [])
        
        # Combine and prioritize
        all_differentiators = differentiators + advantages
        
        return all_differentiators[:4]  # Return top 4 differentiators
    
    def _determine_implementation_priority(self, strategy: Dict[str, Any]) -> List[str]:
        """Determine priority order for implementing strategic elements"""
        return [
            "Core messaging development",
            "Content structure implementation", 
            "Persuasion technique integration",
            "CTA optimization",
            "Competitive positioning refinement"
        ]
    
    def _assess_resource_requirements(self, strategy: Dict[str, Any]) -> Dict[str, str]:
        """Assess resource requirements for strategy implementation"""
        return {
            "copywriting_complexity": "medium",
            "design_requirements": "standard",
            "research_needed": "minimal",
            "approval_complexity": "low",
            "technical_requirements": "basic"
        }
    
    def _suggest_implementation_timeline(self, strategy: Dict[str, Any]) -> Dict[str, str]:
        """Suggest timeline for strategy implementation"""
        return {
            "strategy_finalization": "1 day",
            "content_creation": "2-3 days",
            "review_and_approval": "1-2 days",
            "optimization_setup": "1 day",
            "total_timeline": "5-7 days"
        }
    
    def _define_success_metrics(self, strategy: Dict[str, Any], content_type: str) -> List[str]:
        """Define key success metrics for the strategy"""
        base_metrics = [
            "Conversion rate",
            "Engagement rate",
            "Click-through rate"
        ]
        
        # Content-type specific metrics
        if content_type == "ad":
            base_metrics.extend(["CPM", "CTR", "ROAS"])
        elif content_type == "landing_page":
            base_metrics.extend(["Bounce rate", "Time on page", "Form completion rate"])
        elif content_type == "email":
            base_metrics.extend(["Open rate", "Reply rate", "Unsubscribe rate"])
        
        return base_metrics
    
    def _identify_potential_issues(self, strategy: Dict[str, Any], persona: Dict[str, Any]) -> List[str]:
        """Identify potential issues with the strategy"""
        issues = []
        
        # Check for potential tone mismatches
        strategy_tone = strategy.get("messaging_optimization", {}).get("tone_specification", "").lower()
        if "aggressive" in strategy_tone:
            issues.append("Potentially aggressive tone may alienate sensitive personas")
        
        # Check for complexity issues
        persuasion_techniques = strategy.get("persuasion_techniques", {})
        if len(persuasion_techniques.get("supporting_techniques", [])) > 4:
            issues.append("Too many persuasion techniques may create confusion")
        
        # Add generic risk factors
        issues.extend([
            "Market conditions may affect strategy effectiveness",
            "Competitive responses could require adjustments"
        ])
        
        return issues
    
    def _suggest_risk_mitigation(self, strategy: Dict[str, Any]) -> List[str]:
        """Suggest risk mitigation strategies"""
        return [
            "Implement A/B testing for key strategic elements",
            "Monitor performance metrics closely in first week",
            "Prepare alternative messaging variants",
            "Set up feedback collection from target audience",
            "Plan regular strategy review checkpoints"
        ]
    
    def _develop_contingency_plans(self, strategy: Dict[str, Any]) -> List[str]:
        """Develop contingency plans for strategy issues"""
        return [
            "Alternative messaging framework if primary approach underperforms",
            "Backup persuasion techniques for different persona segments",
            "Simplified content structure for better performance",
            "Emergency competitor response protocols",
            "Quick pivot strategies for changing market conditions"
        ]
    
    # Continue with additional calculation methods...
    def _calculate_messaging_alignment(self, strategy: Dict[str, Any], persona: Dict[str, Any]) -> float:
        """Calculate how well messaging aligns with persona"""
        core_messaging = strategy.get("core_messaging", {})
        persona_goals = persona.get("goals", [])
        persona_pain_points = persona.get("pain_points", [])
        
        # Simple alignment calculation
        alignment_score = 0.7  # Base score
        
        # Check if messaging addresses key persona elements
        messaging_text = str(core_messaging).lower()
        
        # Check goal alignment
        goals_addressed = sum(1 for goal in persona_goals 
                            if any(word in messaging_text for word in str(goal).lower().split()))
        if goals_addressed > 0:
            alignment_score += min(0.15, goals_addressed / len(persona_goals) * 0.3)
        
        # Check pain point alignment  
        pain_points_addressed = sum(1 for pain in persona_pain_points
                                  if any(word in messaging_text for word in str(pain).lower().split()))
        if pain_points_addressed > 0:
            alignment_score += min(0.15, pain_points_addressed / len(persona_pain_points) * 0.3)
        
        return min(alignment_score, 1.0)
    
    def _calculate_persuasion_alignment(self, strategy: Dict[str, Any], persona: Dict[str, Any]) -> float:
        """Calculate persuasion technique alignment with persona psychology"""
        persuasion_techniques = strategy.get("persuasion_techniques", {})
        persona_psychographics = persona.get("psychographics", {})
        
        # Base alignment score
        alignment_score = 0.75
        
        # Check if persuasion techniques match persona characteristics
        primary_technique = persuasion_techniques.get("primary_technique", "").lower()
        psychographic_text = str(persona_psychographics).lower()
        
        # Simple matching heuristics
        if "social" in primary_technique and ("community" in psychographic_text or "social" in psychographic_text):
            alignment_score += 0.1
        
        if "authority" in primary_technique and ("expert" in psychographic_text or "quality" in psychographic_text):
            alignment_score += 0.1
        
        if "urgency" in primary_technique and ("busy" in psychographic_text or "time" in psychographic_text):
            alignment_score += 0.1
        
        return min(alignment_score, 1.0)
    
    def _calculate_tone_alignment(self, strategy: Dict[str, Any], persona: Dict[str, Any]) -> float:
        """Calculate tone alignment between strategy and persona preferences"""
        messaging_optimization = strategy.get("messaging_optimization", {})
        communication_preferences = persona.get("communication_preferences", {})
        
        strategy_tone = messaging_optimization.get("tone_specification", "").lower()
        persona_tone = str(communication_preferences.get("tone", "")).lower()
        
        # Calculate alignment based on tone matching
        if strategy_tone and persona_tone:
            # Simple keyword matching
            strategy_words = set(strategy_tone.split())
            persona_words = set(persona_tone.split())
            
            # Calculate overlap
            overlap = len(strategy_words.intersection(persona_words))
            total_words = len(strategy_words.union(persona_words))
            
            if total_words > 0:
                return min(0.6 + (overlap / total_words) * 0.4, 1.0)
        
        return 0.75  # Default alignment score
