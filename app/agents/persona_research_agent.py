import json
from typing import Dict, Any, List
from datetime import datetime

from app.agents.base_agent import BaseAgent, TaskRequest
from app.models.agent_schemas import AgentCapability, AgentStatus
from app.services.llm_service import LLMService, LLMRequest
from app.core.config import settings
from app.services.persona_service import PersonaService

class PersonaResearchAgent(BaseAgent):
    """
    Specialized agent for comprehensive persona research and demographic analysis.
    
    This agent conducts deep market research to understand target audiences,
    analyzing demographics, psychographics, pain points, and behavioral patterns
    to inform content strategy and messaging decisions.
    """
    
    def __init__(self, llm_service: LLMService):
        capabilities = [
            AgentCapability(
                name="demographic_analysis",
                description="Analyze target audience demographics and characteristics",
                proficiency_level=0.95,
                required_resources=["groq_api"],
                average_execution_time=15.0
            ),
            AgentCapability(
                name="psychographic_profiling",
                description="Understand audience values, interests, and lifestyle patterns",
                proficiency_level=0.90,
                required_resources=["groq_api"],
                average_execution_time=20.0
            ),
            AgentCapability(
                name="pain_point_analysis",
                description="Identify and analyze customer pain points and challenges",
                proficiency_level=0.88,
                required_resources=["groq_api"],
                average_execution_time=12.0
            ),
            AgentCapability(
                name="competitive_research",
                description="Research competitive landscape and messaging approaches",
                proficiency_level=0.85,
                required_resources=["groq_api"],
                average_execution_time=25.0
            )
        ]
        
        super().__init__(
            agent_id="persona_researcher",
            name="Persona Research Specialist",
            role="Market Research & Demographic Analysis",
            llm_service=llm_service,
            capabilities=capabilities
            
        )
        self.persona_service = PersonaService()
        # Persona research templates and methodologies
        self.research_frameworks = {
            "demographic": self._get_demographic_framework(),
            "psychographic": self._get_psychographic_framework(),
            "behavioral": self._get_behavioral_framework(),
            "competitive": self._get_competitive_framework()
        }
        
        self.logger.info("Persona Research Agent initialized with specialized research frameworks")
    
    async def process_task(self, task: TaskRequest) -> Dict[str, Any]:
        """
        Execute comprehensive persona research based on the task requirements.
        
        This method orchestrates different types of research (demographic, psychographic,
        behavioral) to create a complete picture of the target persona.
        """
        self.logger.info(
            "Starting persona research task",
            task_id=task.task_id,
            persona_id=task.input_data.get("persona_id"),
            research_depth=task.input_data.get("research_depth", "standard")
        )
        
        persona_id = task.input_data.get("persona_id")
        additional_context = task.input_data.get("additional_context", "")
        research_depth = task.input_data.get("research_depth", "comprehensive")
        
        if not persona_id:
            raise ValueError("persona_id is required for persona research")
        
        # Build comprehensive research prompt
        research_prompt = self._build_research_prompt(persona_id, additional_context, research_depth)
        
        # Execute research using the most capable model
        llm_request = LLMRequest(
            prompt=research_prompt,
            model=settings.GROQ_MODEL_SMART,  # Use smartest model for research
            temperature=0.3,  # Lower temperature for factual research
            max_tokens=1500,
            system_prompt=self._get_researcher_system_prompt()
        )
        
        try:
            response = await self.llm_service.generate_content(
                prompt=research_prompt,
                model=settings.GROQ_MODEL_SMART,
                temperature=0.3,
                max_tokens=1500,
                task_type="research"
            )
            
            # Parse and structure the research results
            research_data = await self._parse_research_results(response.content, persona_id)
            
            # Enhance with additional analysis
            enhanced_data = await self._enhance_research_data(research_data, persona_id)
            
            # Store research in agent memory for future reference
            self.add_memory(f"persona_research_{persona_id}", enhanced_data, ttl=3600)
            
            result = {
                "persona_research": enhanced_data,
                "research_confidence": self._calculate_research_confidence(enhanced_data),
                "research_methodology": research_depth,
                "data_sources": ["market_analysis", "demographic_studies", "behavioral_insights"],
                "research_timestamp": datetime.utcnow().isoformat(),
                "tokens_used": response.tokens_used,
                "api_calls": 1,
                "processing_notes": [
                    f"Research depth: {research_depth}",
                    f"Context provided: {'Yes' if additional_context else 'No'}",
                    f"Model used: {response.model_used}"
                ]
            }
            
            self.logger.info(
                "Persona research completed successfully",
                task_id=task.task_id,
                confidence=result["research_confidence"],
                data_points=len(enhanced_data.get("key_insights", []))
            )
            
            return result
            
        except Exception as e:
            self.logger.error(
                "Persona research failed",
                task_id=task.task_id,
                error=str(e),
                persona_id=persona_id
            )
            raise
    
    def _build_research_prompt(self, persona_id: str, context: str, depth: str) -> str:
        """
        Build a comprehensive research prompt tailored to the specific persona
        and research requirements.
        """
        base_prompt = f"""
        As a senior market research analyst, conduct comprehensive research on the '{persona_id}' persona.
        
        {"Additional Context: " + context if context else ""}
        
        Research Depth: {depth}
        
        Provide detailed analysis covering:
        
        1. DEMOGRAPHIC PROFILE
        {self.research_frameworks["demographic"]}
        
        2. PSYCHOGRAPHIC ANALYSIS  
        {self.research_frameworks["psychographic"]}
        
        3. BEHAVIORAL PATTERNS
        {self.research_frameworks["behavioral"]}
        
        4. COMPETITIVE LANDSCAPE
        {self.research_frameworks["competitive"]}
        
        Format your response as structured JSON with the following sections:
        - demographics: {{detailed demographic breakdown}}
        - psychographics: {{values, interests, lifestyle}}
        - pain_points: [list of specific challenges]
        - goals: [primary and secondary objectives]
        - digital_behavior: {{online habits and preferences}}
        - communication_preferences: {{tone, style, messaging approach}}
        - competitive_insights: {{what competitors are doing}}
        - market_trends: {{relevant industry developments}}
        - key_insights: [actionable research findings]
        - confidence_indicators: {{factors supporting research accuracy}}
        
        Ensure all insights are specific, actionable, and based on realistic market understanding.
        """
        
        return base_prompt
    
    def _get_researcher_system_prompt(self) -> str:
        """System prompt that defines the researcher's expertise and approach"""
        return """
        You are a world-class market research analyst with 15+ years of experience in consumer behavior, 
        demographic analysis, and market segmentation. Your expertise includes:
        
        - Advanced demographic and psychographic profiling
        - Consumer behavior pattern analysis
        - Competitive landscape assessment
        - Market trend identification and analysis
        - Data-driven insight generation
        
        Your research is always:
        - Evidence-based and grounded in market realities
        - Specific and actionable for marketing teams
        - Comprehensive yet focused on key insights
        - Structured and easy to understand
        - Forward-looking with trend awareness
        
        Approach each research task with scientific rigor while maintaining practical applicability.
        """
    
    async def _parse_research_results(self, raw_response: str, persona_id: str) -> Dict[str, Any]:
        """
        Parse and structure the raw LLM response into organized research data.
        
        Handles JSON parsing with fallback strategies for malformed responses.
        """
        try:
            # Attempt direct JSON parsing
            research_data = json.loads(raw_response)
            
            # Validate required sections
            required_sections = [
                "demographics", "psychographics", "pain_points", 
                "goals", "digital_behavior", "communication_preferences"
            ]
            
            for section in required_sections:
                if section not in research_data:
                    research_data[section] = {}
            
            return research_data
            
        except json.JSONDecodeError:
            # Fallback: Structure raw text into organized format
            self.logger.warning(
                "Failed to parse JSON research response, using text analysis fallback",
                persona_id=persona_id
            )
            
            return await self._extract_structured_data_from_text(raw_response, persona_id)
    
    async def _extract_structured_data_from_text(self, text: str, persona_id: str) -> Dict[str, Any]:
        """
        Extract structured data from unstructured text response using LLM parsing.
        
        This fallback method uses a secondary LLM call to structure the data properly.
        """
        parsing_prompt = f"""
        Extract and structure the following market research data into JSON format:
        
        {text}
        
        Convert this into a JSON object with these exact keys:
        - demographics: object with demographic details
        - psychographics: object with values and interests  
        - pain_points: array of specific challenges
        - goals: array of objectives
        - digital_behavior: object with online habits
        - communication_preferences: object with messaging preferences
        - key_insights: array of actionable findings
        
        Ensure the JSON is valid and complete.
        """
        
        try:
            parsing_response = await self.llm_service.generate_content(
                prompt=parsing_prompt,
                model=settings.GROQ_MODEL_FAST,  # Use faster model for parsing
                temperature=0.1,  # Very low temperature for structured output
                max_tokens=800
            )
            
            return json.loads(parsing_response.content)
            
        except Exception as e:
            self.logger.error("Fallback parsing also failed", error=str(e))
            
            # Ultimate fallback: return basic structure
            return {
                "demographics": {"age_range": "unknown", "analysis": text[:500]},
                "psychographics": {"values": "analysis_needed"},
                "pain_points": ["requires_further_analysis"],
                "goals": ["needs_clarification"],
                "digital_behavior": {"patterns": "to_be_determined"},
                "communication_preferences": {"style": "standard"},
                "key_insights": ["raw_data_needs_processing"],
                "parsing_error": str(e)
            }
    
    async def _enhance_research_data(self, research_data: Dict[str, Any], persona_id: str) -> Dict[str, Any]:
        """
        Enhance the basic research data with additional analysis and insights.
        
        Adds calculated metrics, trend analysis, and actionable recommendations.
        """
        enhanced_data = research_data.copy()
        
        # Add calculated insights
        enhanced_data["calculated_metrics"] = {
            "digital_engagement_score": self._calculate_digital_engagement(research_data),
            "price_sensitivity": self._assess_price_sensitivity(research_data),
            "content_preference_score": self._assess_content_preferences(research_data),
            "trust_factors": self._identify_trust_factors(research_data)
        }
        
        # Add actionable recommendations
        enhanced_data["actionable_recommendations"] = {
            "content_strategy": self._generate_content_recommendations(research_data),
            "messaging_approach": self._generate_messaging_recommendations(research_data),
            "channel_strategy": self._generate_channel_recommendations(research_data),
            "timing_insights": self._generate_timing_recommendations(research_data)
        }
        
        # Add research quality indicators
        enhanced_data["research_quality"] = {
            "completeness_score": self._assess_data_completeness(research_data),
            "specificity_level": self._assess_data_specificity(research_data),
            "actionability_rating": self._assess_actionability(research_data)
        }
        
        return enhanced_data
    
    def _calculate_research_confidence(self, research_data: Dict[str, Any]) -> float:
        """
        Calculate overall confidence score for the research results.
        
        Based on data completeness, specificity, and internal consistency.
        """
        completeness = self._assess_data_completeness(research_data)
        specificity = self._assess_data_specificity(research_data)
        consistency = self._assess_data_consistency(research_data)
        
        # Weighted average with emphasis on completeness
        confidence = (completeness * 0.4 + specificity * 0.3 + consistency * 0.3)
        
        return min(max(confidence, 0.1), 0.95)  # Clamp between 0.1 and 0.95
    
    def _assess_data_completeness(self, data: Dict[str, Any]) -> float:
        """Assess how complete the research data is"""
        required_fields = [
            "demographics", "psychographics", "pain_points", 
            "goals", "digital_behavior", "communication_preferences"
        ]
        
        present_fields = sum(1 for field in required_fields if field in data and data[field])
        return present_fields / len(required_fields)
    
    def _assess_data_specificity(self, data: Dict[str, Any]) -> float:
        """Assess how specific and detailed the research data is"""
        specificity_indicators = 0
        total_checks = 0
        
        # Check demographics specificity
        if "demographics" in data:
            demo = data["demographics"]
            if isinstance(demo.get("age_range"), str) and "-" in demo.get("age_range", ""):
                specificity_indicators += 1
            if demo.get("income"):
                specificity_indicators += 1
            total_checks += 2
        
        # Check pain points specificity
        if "pain_points" in data and isinstance(data["pain_points"], list):
            specificity_indicators += min(len(data["pain_points"]) / 5.0, 1.0)
            total_checks += 1
        
        return specificity_indicators / max(total_checks, 1)
    
    def _assess_data_consistency(self, data: Dict[str, Any]) -> float:
        """Assess internal consistency of the research data"""
        # Simple consistency checks - in production, this could be more sophisticated
        consistency_score = 0.8  # Base consistency assumption
        
        # Check for obvious contradictions
        # This could be enhanced with more sophisticated consistency checking
        
        return consistency_score
    
    # Helper methods for enhancement calculations
    def _calculate_digital_engagement(self, data: Dict[str, Any]) -> float:
        """Calculate digital engagement score based on digital behavior data"""
        digital_behavior = data.get("digital_behavior", {})
        
        # Simple scoring based on presence of digital behavior indicators
        score = 0.5  # Base score
        
        if "social_media" in str(digital_behavior).lower():
            score += 0.2
        if "mobile" in str(digital_behavior).lower():
            score += 0.2
        if "online" in str(digital_behavior).lower():
            score += 0.1
            
        return min(score, 1.0)
    
    def _assess_price_sensitivity(self, data: Dict[str, Any]) -> str:
        """Assess price sensitivity based on demographics and goals"""
        demographics = data.get("demographics", {})
        goals = data.get("goals", [])
        
        # Simple heuristic - could be enhanced with more sophisticated analysis
        income_str = str(demographics.get("income", "")).lower()
        goals_str = str(goals).lower()
        
        if "budget" in goals_str or "save" in goals_str or "affordable" in income_str:
            return "high"
        elif "premium" in goals_str or "quality" in goals_str:
            return "low"
        else:
            return "moderate"
    
    def _assess_content_preferences(self, data: Dict[str, Any]) -> Dict[str, str]:
        """Assess content preferences based on communication and digital behavior"""
        comm_prefs = data.get("communication_preferences", {})
        digital_behavior = data.get("digital_behavior", {})
        
        return {
            "format_preference": "visual" if "visual" in str(comm_prefs).lower() else "text",
            "length_preference": "concise" if "brief" in str(comm_prefs).lower() else "detailed",
            "tone_preference": str(comm_prefs.get("tone", "professional")).lower()
        }
    
    def _identify_trust_factors(self, data: Dict[str, Any]) -> List[str]:
        """Identify key trust factors for the persona"""
        trust_factors = ["expertise", "transparency"]
        
        goals = str(data.get("goals", [])).lower()
        pain_points = str(data.get("pain_points", [])).lower()
        
        if "safety" in pain_points or "security" in pain_points:
            trust_factors.append("security_credentials")
        if "quality" in goals:
            trust_factors.append("quality_assurance")
        if "family" in str(data).lower():
            trust_factors.append("family_testimonials")
            
        return trust_factors
    
    def _generate_content_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Generate content strategy recommendations"""
        recommendations = []
        
        comm_prefs = data.get("communication_preferences", {})
        pain_points = data.get("pain_points", [])
        
        if "simple" in str(comm_prefs).lower():
            recommendations.append("Use clear, jargon-free language")
        
        if len(pain_points) > 0:
            recommendations.append("Address pain points directly in opening")
        
        recommendations.append("Include social proof and testimonials")
        recommendations.append("Use persona-specific examples and scenarios")
        
        return recommendations
    
    def _generate_messaging_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Generate messaging approach recommendations"""
        return [
            "Lead with primary value proposition",
            "Address top 3 pain points explicitly", 
            "Use empathetic and understanding tone",
            "Include clear call-to-action"
        ]
    
    def _generate_channel_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Generate channel strategy recommendations"""
        digital_behavior = data.get("digital_behavior", {})
        
        recommendations = ["Email marketing"]
        
        if "social" in str(digital_behavior).lower():
            recommendations.append("Social media engagement")
        if "mobile" in str(digital_behavior).lower():
            recommendations.append("Mobile-optimized content")
        if "search" in str(digital_behavior).lower():
            recommendations.append("Search engine marketing")
            
        return recommendations
    
    def _generate_timing_recommendations(self, data: Dict[str, Any]) -> List[str]:
        """Generate timing insights"""
        return [
            "Avoid early morning communications",
            "Peak engagement during lunch hours",
            "Weekend communications for family-focused personas"
        ]
    
    def _get_demographic_framework(self) -> str:
        """Research framework for demographic analysis"""
        return """
        - Age range and generational cohort characteristics
        - Income level and spending power analysis  
        - Geographic location and regional preferences
        - Education level and professional background
        - Family status and household composition
        - Life stage and major life transitions
        """
    
    def _get_psychographic_framework(self) -> str:
        """Research framework for psychographic profiling"""  
        return """
        - Core values and belief systems
        - Lifestyle preferences and priorities
        - Interests, hobbies, and passions
        - Personality traits and characteristics
        - Motivations and driving factors
        - Attitudes toward brands and marketing
        """
    
    def _get_behavioral_framework(self) -> str:
        """Research framework for behavioral analysis"""
        return """
        - Digital behavior patterns and platform usage
        - Content consumption habits and preferences
        - Purchase decision-making process
        - Information gathering and research methods
        - Communication channel preferences
        - Response patterns to marketing messages
        """
    
    def _get_competitive_framework(self) -> str:
        """Research framework for competitive analysis"""
        return """
        - Competitor messaging strategies and positioning
        - Successful campaigns targeting this persona
        - Market gaps and underserved needs
        - Pricing strategies and value propositions
        - Channel strategies and media mix
        - Emerging trends and opportunities
        """
