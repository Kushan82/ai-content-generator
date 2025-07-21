import json
import re
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from app.agents.base_agent import BaseAgent, TaskRequest
from app.models.agent_schemas import AgentCapability, AgentStatus
from app.services.llm_service import LLMService, LLMRequest
from app.core.config import settings

class QualityAssuranceAgent(BaseAgent):
    """
    Specialized agent for comprehensive content quality assurance and optimization.
    
    This agent performs multi-dimensional quality assessment, identifies improvement
    opportunities, ensures brand compliance, and validates content against strategic
    objectives and persona requirements.
    """
    
    def __init__(self, llm_service: LLMService):
        capabilities = [
            AgentCapability(
                name="content_quality_assessment",
                description="Comprehensive quality evaluation across multiple dimensions",
                proficiency_level=0.96,
                required_resources=["groq_api", "quality_frameworks"],
                average_execution_time=22.0
            ),
            AgentCapability(
                name="strategic_alignment_validation",
                description="Validate content adherence to strategic objectives",
                proficiency_level=0.94,
                required_resources=["groq_api", "strategy_data"],
                average_execution_time=18.0
            ),
            AgentCapability(
                name="persona_compliance_check",
                description="Ensure content perfectly matches target persona requirements",
                proficiency_level=0.92,
                required_resources=["groq_api", "persona_data"],
                average_execution_time=16.0
            ),
            AgentCapability(
                name="brand_consistency_audit",
                description="Validate brand voice and messaging consistency",
                proficiency_level=0.90,
                required_resources=["groq_api", "brand_guidelines"],
                average_execution_time=14.0
            ),
            AgentCapability(
                name="conversion_optimization_analysis",
                description="Analyze and optimize content for conversion performance",
                proficiency_level=0.89,
                required_resources=["groq_api", "conversion_data"],
                average_execution_time=20.0
            ),
            AgentCapability(
                name="content_enhancement_recommendations",
                description="Provide specific, actionable improvement recommendations",
                proficiency_level=0.91,
                required_resources=["groq_api"],
                average_execution_time=15.0
            )
        ]
        
        super().__init__(
            agent_id="quality_assurance",
            name="Quality Assurance Specialist", 
            role="Content Quality Control & Optimization",
            llm_service=llm_service,
            capabilities=capabilities
        )
        
        # Quality assessment frameworks
        self.quality_frameworks = {
            "clarity": self._get_clarity_framework(),
            "persuasion": self._get_persuasion_framework(), 
            "engagement": self._get_engagement_framework(),
            "conversion": self._get_conversion_framework(),
            "compliance": self._get_compliance_framework()
        }
        
        # Quality scoring weights by content type
        self.scoring_weights = {
            "ad": {"clarity": 0.15, "persuasion": 0.30, "engagement": 0.25, "conversion": 0.30},
            "landing_page": {"clarity": 0.20, "persuasion": 0.25, "engagement": 0.20, "conversion": 0.35},
            "blog_intro": {"clarity": 0.25, "persuasion": 0.20, "engagement": 0.35, "conversion": 0.20},
            "email": {"clarity": 0.20, "persuasion": 0.25, "engagement": 0.30, "conversion": 0.25},
            "social_media": {"clarity": 0.15, "persuasion": 0.20, "engagement": 0.45, "conversion": 0.20}
        }
        
        # Quality benchmarks and thresholds
        self.quality_thresholds = {
            "excellent": 0.90,
            "good": 0.75,
            "acceptable": 0.60,
            "needs_improvement": 0.45,
            "poor": 0.30
        }
        
        self.logger.info("Quality Assurance Agent initialized with comprehensive evaluation frameworks")
    
    async def process_task(self, task: TaskRequest) -> Dict[str, Any]:
        """
        Execute comprehensive quality assurance evaluation of generated content.
        
        This method performs multi-dimensional quality assessment, validates strategic
        alignment, and provides detailed optimization recommendations.
        """
        self.logger.info(
            "Starting quality assurance evaluation",
            task_id=task.task_id,
            content_length=len(str(task.input_data.get("generated_content", ""))),
            content_type=task.input_data.get("content_type")
        )
        
        # Extract input data
        generated_content = task.input_data.get("generated_content", "")
        content_type = task.input_data.get("content_type", "")
        persona_research = task.input_data.get("persona_research", {})
        content_strategy = task.input_data.get("content_strategy", {})
        brand_guidelines = task.input_data.get("brand_guidelines", {})
        business_objectives = task.input_data.get("business_objectives", [])
        
        if not generated_content:
            raise ValueError("generated_content is required for quality assurance")
        
        # Perform comprehensive quality assessment
        quality_scores = await self._perform_quality_assessment(
            generated_content, content_type, persona_research, content_strategy
        )
        
        # Validate strategic alignment
        strategic_validation = await self._validate_strategic_alignment(
            generated_content, content_strategy, business_objectives
        )
        
        # Check persona compliance
        persona_compliance = await self._check_persona_compliance(
            generated_content, persona_research, content_type
        )
        
        # Audit brand consistency
        brand_audit = await self._audit_brand_consistency(
            generated_content, brand_guidelines, content_type
        )
        
        # Analyze conversion optimization potential
        conversion_analysis = await self._analyze_conversion_optimization(
            generated_content, content_type, persona_research
        )
        
        # Generate specific improvement recommendations
        improvement_recommendations = await self._generate_improvement_recommendations(
            generated_content, quality_scores, strategic_validation, persona_compliance
        )
        
        # Calculate overall quality score
        overall_quality_score = self._calculate_overall_quality_score(
            quality_scores, content_type
        )
        
        # Generate optimized version if needed
        optimized_content = None
        if overall_quality_score < self.quality_thresholds["good"]:
            optimized_content = await self._generate_optimized_version(
                generated_content, improvement_recommendations, content_strategy
            )
        
        # Compile comprehensive quality report
        result = {
            "quality_assessment": {
                "overall_score": overall_quality_score,
                "quality_level": self._determine_quality_level(overall_quality_score),
                "dimension_scores": quality_scores,
                "scoring_breakdown": self._get_scoring_breakdown(quality_scores, content_type)
            },
            "strategic_validation": strategic_validation,
            "persona_compliance": persona_compliance,
            "brand_consistency": brand_audit,
            "conversion_analysis": conversion_analysis,
            "improvement_recommendations": improvement_recommendations,
            "optimized_content": optimized_content,
            "compliance_status": {
                "strategic_compliance": strategic_validation["overall_compliance"],
                "persona_compliance": persona_compliance["compliance_score"],
                "brand_compliance": brand_audit["consistency_score"],
                "minimum_standards_met": overall_quality_score >= self.quality_thresholds["acceptable"]
            },
            "performance_predictions": self._predict_performance_metrics(
                generated_content, quality_scores, persona_compliance
            ),
            "risk_assessment": self._assess_content_risks(
                generated_content, persona_research, brand_guidelines
            ),
            "optimization_priority": self._determine_optimization_priority(
                overall_quality_score, improvement_recommendations
            ),
            "tokens_used": 0,  # Will be updated with actual usage
            "api_calls": 0,    # Will be updated with actual calls
            "qa_metadata": {
                "content_type": content_type,
                "content_word_count": len(generated_content.split()),
                "evaluation_timestamp": datetime.utcnow().isoformat(),
                "quality_frameworks_used": list(self.quality_frameworks.keys()),
                "assessment_confidence": self._calculate_assessment_confidence(quality_scores)
            }
        }
        
        self.logger.info(
            "Quality assurance evaluation completed",
            task_id=task.task_id,
            overall_score=overall_quality_score,
            quality_level=result["quality_assessment"]["quality_level"],
            needs_optimization=optimized_content is not None
        )
        
        return result
    
    async def _perform_quality_assessment(
        self,
        content: str,
        content_type: str,
        persona_research: Dict[str, Any],
        content_strategy: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Perform comprehensive quality assessment across multiple dimensions.
        """
        assessment_prompt = f"""
        As a senior content quality assurance specialist, evaluate this {content_type} content 
        across multiple quality dimensions:
        
        CONTENT TO EVALUATE:
        {content}
        
        EVALUATION CONTEXT:
        Content Type: {content_type}
        Target Persona: {persona_research.get('demographics', {})}
        Strategic Context: {content_strategy.get('core_messaging', {})}
        
        QUALITY FRAMEWORKS:
        {self.quality_frameworks["clarity"]}
        {self.quality_frameworks["persuasion"]}
        {self.quality_frameworks["engagement"]}
        {self.quality_frameworks["conversion"]}
        
        Evaluate the content and provide scores (0.0-1.0) for each dimension:
        
        Response format (JSON only):
        {{
            "clarity_score": 0.0-1.0,
            "persuasion_score": 0.0-1.0,
            "engagement_score": 0.0-1.0,
            "conversion_score": 0.0-1.0,
            "detailed_analysis": {{
                "clarity_factors": ["specific clarity strengths and weaknesses"],
                "persuasion_elements": ["persuasion techniques identified"],
                "engagement_drivers": ["engagement factors present"],
                "conversion_barriers": ["obstacles to conversion"]
            }},
            "improvement_areas": ["specific areas needing enhancement"]
        }}
        
        Be objective, specific, and provide actionable insights.
        """
        
        try:
            assessment_response = await self.llm_service.generate_content(
                prompt=assessment_prompt,
                model=settings.GROQ_MODEL_SMART,
                temperature=0.2,  # Low temperature for consistent evaluation
                max_tokens=1000
            )
            
            assessment_data = json.loads(assessment_response.content)
            
            # Extract and validate scores
            quality_scores = {
                "clarity": max(0.0, min(1.0, assessment_data.get("clarity_score", 0.5))),
                "persuasion": max(0.0, min(1.0, assessment_data.get("persuasion_score", 0.5))),
                "engagement": max(0.0, min(1.0, assessment_data.get("engagement_score", 0.5))),
                "conversion": max(0.0, min(1.0, assessment_data.get("conversion_score", 0.5)))
            }
            
            # Store detailed analysis for reference
            self.add_memory(
                f"quality_analysis_{hash(content[:100])}",
                assessment_data.get("detailed_analysis", {}),
                ttl=3600
            )
            
            return quality_scores
            
        except Exception as e:
            self.logger.error(f"Quality assessment failed, using fallback scoring: {e}")
            return await self._fallback_quality_assessment(content, content_type)
    
    async def _fallback_quality_assessment(self, content: str, content_type: str) -> Dict[str, float]:
        """
        Fallback quality assessment using rule-based evaluation.
        """
        scores = {}
        content_lower = content.lower()
        word_count = len(content.split())
        
        # Clarity assessment
        clarity_factors = []
        if word_count >= 50:  # Adequate length
            clarity_factors.append(0.2)
        if any(word in content_lower for word in ["clear", "simple", "easy", "understand"]):
            clarity_factors.append(0.3)
        if len([s for s in content.split('.') if len(s.split()) <= 20]) / max(len(content.split('.')), 1) > 0.7:
            clarity_factors.append(0.2)  # Reasonable sentence length
        scores["clarity"] = min(sum(clarity_factors), 1.0) if clarity_factors else 0.5
        
        # Persuasion assessment
        persuasion_indicators = ["benefit", "advantage", "proven", "guarantee", "exclusive", "limited"]
        persuasion_count = sum(1 for word in persuasion_indicators if word in content_lower)
        scores["persuasion"] = min(persuasion_count / 4.0, 1.0) + 0.3  # Base score + indicators
        
        # Engagement assessment
        engagement_indicators = ["you", "your", "discover", "imagine", "?", "!"]
        engagement_count = sum(content_lower.count(word) for word in engagement_indicators[:4])
        engagement_count += content.count("?") + content.count("!")
        scores["engagement"] = min(engagement_count / 8.0, 0.8) + 0.2  # Base + indicators
        
        # Conversion assessment
        cta_indicators = ["click", "get", "start", "try", "join", "buy", "learn", "discover"]
        cta_count = sum(1 for word in cta_indicators if word in content_lower)
        scores["conversion"] = min(cta_count / 3.0, 0.7) + 0.3  # Base + CTA strength
        
        return scores
    
    async def _validate_strategic_alignment(
        self,
        content: str,
        content_strategy: Dict[str, Any],
        business_objectives: List[str]
    ) -> Dict[str, Any]:
        """
        Validate how well content aligns with strategic objectives and requirements.
        """
        validation_prompt = f"""
        Validate this content's alignment with the provided strategic requirements:
        
        CONTENT:
        {content}
        
        STRATEGIC REQUIREMENTS:
        Core Messaging: {content_strategy.get('core_messaging', {})}
        Persuasion Strategy: {content_strategy.get('persuasion_techniques', {})}
        Call-to-Action Strategy: {content_strategy.get('call_to_action', {})}
        Messaging Optimization: {content_strategy.get('messaging_optimization', {})}
        
        Business Objectives: {', '.join(business_objectives) if business_objectives else 'General conversion'}
        
        Evaluate alignment and provide specific feedback:
        
        Response format (JSON):
        {{
            "overall_compliance": 0.0-1.0,
            "messaging_alignment": 0.0-1.0,
            "persuasion_alignment": 0.0-1.0,
            "cta_alignment": 0.0-1.0,
            "tone_alignment": 0.0-1.0,
            "strategic_gaps": ["specific gaps or misalignments"],
            "alignment_strengths": ["areas of strong strategic adherence"],
            "strategic_recommendations": ["specific improvements for better alignment"]
        }}
        """
        
        try:
            validation_response = await self.llm_service.generate_content(
                prompt=validation_prompt,
                model=settings.GROQ_MODEL_SMART,
                temperature=0.2,
                max_tokens=800
            )
            
            return json.loads(validation_response.content)
            
        except Exception as e:
            self.logger.error(f"Strategic validation failed, using fallback: {e}")
            return self._fallback_strategic_validation(content, content_strategy)
    
    def _fallback_strategic_validation(self, content: str, content_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback strategic validation using rule-based analysis."""
        content_lower = content.lower()
        
        # Check messaging alignment
        core_messaging = content_strategy.get("core_messaging", {})
        primary_value_prop = str(core_messaging.get("primary_value_proposition", "")).lower()
        
        messaging_score = 0.5  # Base score
        if primary_value_prop:
            value_words = [word for word in primary_value_prop.split() if len(word) > 3]
            words_found = sum(1 for word in value_words if word in content_lower)
            if value_words:
                messaging_score += min(0.4, (words_found / len(value_words)) * 0.5)
        
        # Check CTA alignment
        cta_strategy = content_strategy.get("call_to_action", {})
        primary_cta = str(cta_strategy.get("primary_cta", "")).lower()
        cta_score = 0.6 if any(word in content_lower for word in primary_cta.split()) else 0.3
        
        # Overall compliance
        overall_compliance = (messaging_score + cta_score) / 2
        
        return {
            "overall_compliance": overall_compliance,
            "messaging_alignment": messaging_score,
            "persuasion_alignment": 0.7,  # Default
            "cta_alignment": cta_score,
            "tone_alignment": 0.7,  # Default
            "strategic_gaps": ["Automated analysis - detailed review recommended"],
            "alignment_strengths": ["Basic strategic elements present"],
            "strategic_recommendations": ["Consider manual strategic review for optimization"]
        }
    
    async def _check_persona_compliance(
        self,
        content: str,
        persona_research: Dict[str, Any],
        content_type: str
    ) -> Dict[str, Any]:
        """
        Check how well content complies with target persona characteristics and preferences.
        """
        compliance_prompt = f"""
        Evaluate this content's compliance with the target persona profile:
        
        CONTENT:
        {content}
        
        PERSONA PROFILE:
        Demographics: {persona_research.get('demographics', {})}
        Pain Points: {persona_research.get('pain_points', [])}
        Goals: {persona_research.get('goals', [])}
        Communication Preferences: {persona_research.get('communication_preferences', {})}
        Digital Behavior: {persona_research.get('digital_behavior', {})}
        
        Evaluate persona alignment:
        
        Response format (JSON):
        {{
            "compliance_score": 0.0-1.0,
            "pain_point_addressing": 0.0-1.0,
            "goal_alignment": 0.0-1.0,
            "tone_appropriateness": 0.0-1.0,
            "language_level_fit": 0.0-1.0,
            "persona_specific_elements": ["elements that specifically resonate with this persona"],
            "persona_mismatches": ["aspects that don't align with persona preferences"],
            "persona_optimization_suggestions": ["specific ways to better align with persona"]
        }}
        """
        
        try:
            compliance_response = await self.llm_service.generate_content(
                prompt=compliance_prompt,
                model=settings.GROQ_MODEL_SMART,
                temperature=0.2,
                max_tokens=700
            )
            
            return json.loads(compliance_response.content)
            
        except Exception as e:
            self.logger.error(f"Persona compliance check failed, using fallback: {e}")
            return self._fallback_persona_compliance(content, persona_research)
    
    def _fallback_persona_compliance(self, content: str, persona_research: Dict[str, Any]) -> Dict[str, Any]:
        """Fallback persona compliance check using rule-based analysis."""
        content_lower = content.lower()
        
        # Check pain point addressing
        pain_points = persona_research.get("pain_points", [])
        pain_point_score = 0.0
        if pain_points:
            addressed_points = 0
            for pain_point in pain_points[:3]:  # Check top 3
                pain_words = [word.lower() for word in str(pain_point).split() if len(word) > 3]
                if any(word in content_lower for word in pain_words):
                    addressed_points += 1
            pain_point_score = addressed_points / min(len(pain_points), 3)
        
        # Check communication style alignment
        comm_prefs = persona_research.get("communication_preferences", {})
        tone = str(comm_prefs.get("tone", "")).lower()
        
        tone_score = 0.7  # Default
        if "casual" in tone and any(word in content_lower for word in ["you", "your", "we"]):
            tone_score = 0.8
        elif "professional" in tone and not any(word in content_lower for word in ["hey", "awesome", "cool"]):
            tone_score = 0.8
        
        compliance_score = (pain_point_score * 0.4 + tone_score * 0.3 + 0.3)  # 0.3 for base compliance
        
        return {
            "compliance_score": min(compliance_score, 1.0),
            "pain_point_addressing": pain_point_score,
            "goal_alignment": 0.7,  # Default
            "tone_appropriateness": tone_score,
            "language_level_fit": 0.7,  # Default
            "persona_specific_elements": ["Basic persona elements identified"],
            "persona_mismatches": [],
            "persona_optimization_suggestions": ["Consider deeper persona-specific customization"]
        }
    
    async def _audit_brand_consistency(
        self,
        content: str,
        brand_guidelines: Dict[str, Any],
        content_type: str
    ) -> Dict[str, Any]:
        """
        Audit content for brand voice and messaging consistency.
        """
        if not brand_guidelines:
            return {
                "consistency_score": 0.8,  # Default when no guidelines provided
                "voice_alignment": 0.8,
                "tone_consistency": 0.8,
                "messaging_consistency": 0.8,
                "brand_compliance_issues": [],
                "brand_strengths": ["No specific brand guidelines to evaluate against"],
                "brand_recommendations": ["Consider providing brand guidelines for more detailed analysis"]
            }
        
        audit_prompt = f"""
        Audit this content against brand guidelines for consistency:
        
        CONTENT:
        {content}
        
        BRAND GUIDELINES:
        Brand Voice: {brand_guidelines.get('voice', {})}
        Brand Tone: {brand_guidelines.get('tone', {})}
        Brand Values: {brand_guidelines.get('values', [])}
        Messaging Guidelines: {brand_guidelines.get('messaging', {})}
        Content Standards: {brand_guidelines.get('standards', {})}
        
        Evaluate brand consistency:
        
        Response format (JSON):
        {{
            "consistency_score": 0.0-1.0,
            "voice_alignment": 0.0-1.0,
            "tone_consistency": 0.0-1.0,
            "messaging_consistency": 0.0-1.0,
            "values_alignment": 0.0-1.0,
            "brand_compliance_issues": ["specific brand guideline violations"],
            "brand_strengths": ["areas of strong brand alignment"],
            "brand_recommendations": ["specific suggestions for better brand consistency"]
        }}
        """
        
        try:
            audit_response = await self.llm_service.generate_content(
                prompt=audit_prompt,
                model=settings.GROQ_MODEL_SMART,
                temperature=0.1,  # Very low temperature for consistency evaluation
                max_tokens=600
            )
            
            return json.loads(audit_response.content)
            
        except Exception as e:
            self.logger.error(f"Brand audit failed, using default scores: {e}")
            return {
                "consistency_score": 0.75,
                "voice_alignment": 0.75,
                "tone_consistency": 0.75,
                "messaging_consistency": 0.75,
                "values_alignment": 0.75,
                "brand_compliance_issues": ["Unable to perform detailed brand analysis"],
                "brand_strengths": ["General brand consistency maintained"],
                "brand_recommendations": ["Manual brand review recommended"]
            }
    
    async def _analyze_conversion_optimization(
        self,
        content: str,
        content_type: str,
        persona_research: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze content for conversion optimization opportunities.
        """
        conversion_prompt = f"""
        Analyze this {content_type} content for conversion optimization:
        
        CONTENT:
        {content}
        
        PERSONA CONTEXT:
        {persona_research.get('goals', [])}
        {persona_research.get('pain_points', [])}
        
        Evaluate conversion elements:
        
        Response format (JSON):
        {{
            "conversion_score": 0.0-1.0,
            "cta_effectiveness": 0.0-1.0,
            "value_proposition_clarity": 0.0-1.0,
            "urgency_presence": 0.0-1.0,
            "trust_signals": 0.0-1.0,
            "conversion_barriers": ["obstacles that might prevent conversion"],
            "conversion_enhancers": ["elements that support conversion"],
            "optimization_opportunities": ["specific ways to improve conversion potential"]
        }}
        """
        
        try:
            conversion_response = await self.llm_service.generate_content(
                prompt=conversion_prompt,
                model=settings.GROQ_MODEL_SMART,
                temperature=0.2,
                max_tokens=600
            )
            
            return json.loads(conversion_response.content)
            
        except Exception as e:
            self.logger.error(f"Conversion analysis failed, using fallback: {e}")
            return self._fallback_conversion_analysis(content)
    
    def _fallback_conversion_analysis(self, content: str) -> Dict[str, Any]:
        """Fallback conversion analysis using rule-based evaluation."""
        content_lower = content.lower()
        
        # CTA effectiveness
        cta_indicators = ["click", "get", "start", "try", "join", "buy", "learn", "discover", "sign up"]
        cta_score = min(sum(1 for word in cta_indicators if word in content_lower) / 3.0, 1.0)
        
        # Value proposition clarity
        value_indicators = ["benefit", "advantage", "solution", "result", "outcome", "transform"]
        value_score = min(sum(1 for word in value_indicators if word in content_lower) / 4.0, 1.0) + 0.2
        
        # Urgency presence
        urgency_indicators = ["now", "today", "limited", "exclusive", "hurry", "soon", "deadline"]
        urgency_score = min(sum(1 for word in urgency_indicators if word in content_lower) / 2.0, 1.0)
        
        # Trust signals
        trust_indicators = ["guarantee", "proven", "secure", "trusted", "certified", "testimonial"]
        trust_score = min(sum(1 for word in trust_indicators if word in content_lower) / 3.0, 1.0) + 0.3
        
        conversion_score = (cta_score * 0.3 + value_score * 0.3 + urgency_score * 0.2 + trust_score * 0.2)
        
        return {
            "conversion_score": min(conversion_score, 1.0),
            "cta_effectiveness": cta_score,
            "value_proposition_clarity": value_score,
            "urgency_presence": urgency_score,
            "trust_signals": trust_score,
            "conversion_barriers": ["Automated analysis - manual review recommended"],
            "conversion_enhancers": ["Basic conversion elements present"],
            "optimization_opportunities": ["Consider A/B testing different approaches"]
        }
    
    async def _generate_improvement_recommendations(
        self,
        content: str,
        quality_scores: Dict[str, float],
        strategic_validation: Dict[str, Any],
        persona_compliance: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """
        Generate specific, actionable improvement recommendations.
        """
        recommendations = {
            "high_priority": [],
            "medium_priority": [],
            "low_priority": [],
            "quick_wins": [],
            "strategic_improvements": []
        }
        
        # High priority recommendations based on quality scores
        for dimension, score in quality_scores.items():
            if score < 0.6:
                if dimension == "clarity":
                    recommendations["high_priority"].append("Simplify language and improve message clarity")
                    recommendations["quick_wins"].append("Break long sentences into shorter, more digestible ones")
                elif dimension == "persuasion":
                    recommendations["high_priority"].append("Strengthen persuasive elements and value proposition")
                    recommendations["strategic_improvements"].append("Integrate more compelling proof points")
                elif dimension == "engagement":
                    recommendations["high_priority"].append("Increase engagement through interactive elements")
                    recommendations["quick_wins"].append("Add questions or direct address to reader")
                elif dimension == "conversion":
                    recommendations["high_priority"].append("Optimize call-to-action and conversion elements")
                    recommendations["strategic_improvements"].append("Create stronger urgency and value clarity")
        
        # Strategic alignment recommendations
        if strategic_validation["overall_compliance"] < 0.7:
            recommendations["medium_priority"].append("Improve alignment with strategic messaging framework")
            if strategic_validation.get("messaging_alignment", 0) < 0.6:
                recommendations["strategic_improvements"].append("Better integrate core value proposition")
        
        # Persona compliance recommendations
        if persona_compliance["compliance_score"] < 0.7:
            recommendations["medium_priority"].append("Enhance persona-specific customization")
            if persona_compliance.get("pain_point_addressing", 0) < 0.6:
                recommendations["high_priority"].append("Directly address target persona's key pain points")
        
        # Content-specific recommendations
        word_count = len(content.split())
        if word_count < 30:
            recommendations["quick_wins"].append("Expand content to provide more value and context")
        elif word_count > 300:
            recommendations["medium_priority"].append("Consider condensing for better readability")
        
        if "?" not in content:
            recommendations["quick_wins"].append("Add engaging questions to increase reader involvement")
        
        if content.count("!") == 0:
            recommendations["low_priority"].append("Consider adding exclamation points for emotional impact")
        
        return recommendations
    
    async def _generate_optimized_version(
        self,
        original_content: str,
        recommendations: Dict[str, List[str]],
        content_strategy: Dict[str, Any]
    ) -> str:
        """
        Generate an optimized version of the content based on QA recommendations.
        """
        optimization_prompt = f"""
        Optimize this content based on the quality assessment recommendations:
        
        ORIGINAL CONTENT:
        {original_content}
        
        IMPROVEMENT RECOMMENDATIONS:
        High Priority: {', '.join(recommendations.get('high_priority', []))}
        Quick Wins: {', '.join(recommendations.get('quick_wins', []))}
        Strategic Improvements: {', '.join(recommendations.get('strategic_improvements', []))}
        
        STRATEGIC CONTEXT:
        {content_strategy.get('core_messaging', {})}
        
        Create an improved version that:
        1. Maintains the original structure and length
        2. Implements the high-priority recommendations
        3. Incorporates quick wins for immediate improvement
        4. Enhances strategic alignment
        5. Improves overall quality while preserving the core message
        
        Return only the optimized content, no explanation needed.
        """
        
        try:
            optimization_response = await self.llm_service.generate_content(
                prompt=optimization_prompt,
                model=settings.GROQ_MODEL_SMART,
                temperature=0.3,
                max_tokens=800
            )
            
            optimized_content = optimization_response.content.strip()
            
            # Validate optimization didn't break the content
            if len(optimized_content) < len(original_content) * 0.5:
                self.logger.warning("Optimization resulted in significantly shorter content")
                return original_content
            
            return optimized_content
            
        except Exception as e:
            self.logger.error(f"Content optimization failed: {e}")
            return original_content
    
    def _calculate_overall_quality_score(self, quality_scores: Dict[str, float], content_type: str) -> float:
        """
        Calculate weighted overall quality score based on content type.
        """
        weights = self.scoring_weights.get(content_type, self.scoring_weights["ad"])
        
        overall_score = sum(
            quality_scores.get(dimension, 0.5) * weight
            for dimension, weight in weights.items()
        )
        
        return min(max(overall_score, 0.0), 1.0)
    
    def _determine_quality_level(self, score: float) -> str:
        """Determine quality level based on score."""
        for level, threshold in sorted(self.quality_thresholds.items(), key=lambda x: x[1], reverse=True):
            if score >= threshold:
                return level
        return "poor"
    
    def _get_scoring_breakdown(self, quality_scores: Dict[str, float], content_type: str) -> Dict[str, Any]:
        """Get detailed scoring breakdown."""
        weights = self.scoring_weights.get(content_type, self.scoring_weights["ad"])
        
        breakdown = {}
        for dimension, score in quality_scores.items():
            weight = weights.get(dimension, 0.25)
            breakdown[dimension] = {
                "score": score,
                "weight": weight,
                "weighted_contribution": score * weight,
                "performance_level": self._determine_quality_level(score)
            }
        
        return breakdown
    
    def _predict_performance_metrics(
        self,
        content: str,
        quality_scores: Dict[str, float],
        persona_compliance: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Predict expected performance metrics based on quality assessment."""
        
        # Base predictions on quality scores and compliance
        overall_quality = sum(quality_scores.values()) / len(quality_scores)
        persona_alignment = persona_compliance.get("compliance_score", 0.7)
        
        # Performance prediction model (simplified)
        engagement_prediction = (quality_scores.get("engagement", 0.5) * 0.6 + 
                               persona_alignment * 0.4)
        
        conversion_prediction = (quality_scores.get("conversion", 0.5) * 0.7 +
                               quality_scores.get("persuasion", 0.5) * 0.3)
        
        return {
            "predicted_engagement_rate": {
                "score": engagement_prediction,
                "level": "high" if engagement_prediction >= 0.8 else "medium" if engagement_prediction >= 0.6 else "low",
                "confidence": "moderate"
            },
            "predicted_conversion_rate": {
                "score": conversion_prediction,
                "level": "high" if conversion_prediction >= 0.8 else "medium" if conversion_prediction >= 0.6 else "low",
                "confidence": "moderate"
            },
            "overall_performance_outlook": {
                "score": (engagement_prediction + conversion_prediction) / 2,
                "outlook": "positive" if overall_quality >= 0.75 else "moderate" if overall_quality >= 0.6 else "needs_improvement"
            }
        }
    
    def _assess_content_risks(
        self,
        content: str,
        persona_research: Dict[str, Any],
        brand_guidelines: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Assess potential risks and issues with the content."""
        risks = {
            "high_risk": [],
            "medium_risk": [],
            "low_risk": []
        }
        
        content_lower = content.lower()
        
        # Check for potential offensive or inappropriate language
        sensitive_words = ["cheap", "free", "spam", "scam", "guaranteed money"]
        if any(word in content_lower for word in sensitive_words):
            risks["medium_risk"].append("Contains potentially problematic language")
        
        # Check for overly aggressive sales language
        aggressive_words = ["must buy", "act now", "don't miss", "hurry"]
        if sum(1 for word in aggressive_words if word in content_lower) >= 2:
            risks["medium_risk"].append("May be perceived as overly aggressive")
        
        # Check for persona misalignment risks
        persona_pain_points = persona_research.get("pain_points", [])
        if not any(any(word in content_lower for word in str(pain).lower().split()) 
                  for pain in persona_pain_points[:3]):
            risks["high_risk"].append("Doesn't address key persona pain points")
        
        # Check for missing call-to-action
        cta_indicators = ["click", "get", "start", "try", "join", "contact", "learn"]
        if not any(word in content_lower for word in cta_indicators):
            risks["high_risk"].append("Missing clear call-to-action")
        
        # Check content length appropriateness
        word_count = len(content.split())
        if word_count < 20:
            risks["medium_risk"].append("Content may be too short to be effective")
        elif word_count > 500:
            risks["low_risk"].append("Content length may reduce engagement")
        
        return risks
    
    def _determine_optimization_priority(
        self,
        quality_score: float,
        recommendations: Dict[str, List[str]]
    ) -> str:
        """Determine optimization priority level."""
        high_priority_count = len(recommendations.get("high_priority", []))
        
        if quality_score < 0.6 or high_priority_count >= 3:
            return "urgent"
        elif quality_score < 0.75 or high_priority_count >= 1:
            return "high"
        elif quality_score < 0.85:
            return "medium"
        else:
            return "low"
    
    def _calculate_assessment_confidence(self, quality_scores: Dict[str, float]) -> float:
        """Calculate confidence in the quality assessment."""
        # Simple confidence calculation based on score variance
        scores = list(quality_scores.values())
        if not scores:
            return 0.5
        
        score_variance = sum((score - sum(scores)/len(scores))**2 for score in scores) / len(scores)
        
        # Higher variance = lower confidence
        confidence = max(0.3, 1.0 - score_variance)
        
        return confidence
    
    # Quality framework definitions
    def _get_clarity_framework(self) -> str:
        return """
        CLARITY EVALUATION FRAMEWORK:
        - Message clarity and comprehension ease
        - Language appropriateness for target audience
        - Sentence structure and readability
        - Logical flow and organization
        - Terminology and jargon usage
        """
    
    def _get_persuasion_framework(self) -> str:
        return """
        PERSUASION EVALUATION FRAMEWORK:
        - Value proposition strength and clarity
        - Proof points and credibility elements
        - Emotional triggers and psychological appeals
        - Objection handling and risk mitigation
        - Persuasion technique integration and effectiveness
        """
    
    def _get_engagement_framework(self) -> str:
        return """
        ENGAGEMENT EVALUATION FRAMEWORK:
        - Reader attention capture and maintenance
        - Personal relevance and connection
        - Interactive elements and questions
        - Emotional resonance and impact
        - Curiosity and interest generation
        """
    
    def _get_conversion_framework(self) -> str:
        return """
        CONVERSION EVALUATION FRAMEWORK:
        - Call-to-action clarity and strength
        - Conversion path simplicity
        - Urgency and motivation creation
        - Trust and credibility establishment
        - Barrier removal and friction reduction
        """
    
    def _get_compliance_framework(self) -> str:
        return """
        COMPLIANCE EVALUATION FRAMEWORK:
        - Brand voice and tone consistency
        - Strategic messaging alignment
        - Persona requirement adherence
        - Legal and regulatory compliance
        - Industry standard conformance
        """
