import json
import re
from typing import Dict, Any, List, Optional
from datetime import datetime

from app.agents.base_agent import BaseAgent, TaskRequest
from app.models.agent_schemas import AgentCapability, AgentStatus
from app.services.llm_service import LLMService, LLMRequest
from app.core.config import settings

class CreativeGenerationAgent(BaseAgent):
    """
    Specialized agent for creative content generation based on strategic blueprints.
    
    This agent transforms persona research and content strategies into compelling,
    persuasive marketing content that aligns with brand voice, audience expectations,
    and business objectives while maintaining creative excellence.
    """
    
    def __init__(self, llm_service: LLMService):
        capabilities = [
            AgentCapability(
                name="creative_copywriting",
                description="Generate compelling marketing copy across multiple formats",
                proficiency_level=0.94,
                required_resources=["groq_api", "content_strategy"],
                average_execution_time=20.0
            ),
            AgentCapability(
                name="brand_voice_adaptation",
                description="Adapt content to match specific brand voices and tones",
                proficiency_level=0.91,
                required_resources=["groq_api", "brand_guidelines"],
                average_execution_time=15.0
            ),
            AgentCapability(
                name="persuasive_writing",
                description="Apply advanced persuasion techniques in content creation",
                proficiency_level=0.93,
                required_resources=["groq_api", "persuasion_strategy"],
                average_execution_time=18.0
            ),
            AgentCapability(
                name="multi_format_adaptation",
                description="Create content optimized for different formats and channels",
                proficiency_level=0.89,
                required_resources=["groq_api", "format_specifications"],
                average_execution_time=12.0
            ),
            AgentCapability(
                name="creative_optimization",
                description="Optimize content for engagement and conversion performance",
                proficiency_level=0.87,
                required_resources=["groq_api", "performance_data"],
                average_execution_time=16.0
            )
        ]
        
        super().__init__(
            agent_id="creative_generator",
            name="Creative Content Generator",
            role="Creative Content Development & Copywriting Excellence",
            llm_service=llm_service,
            capabilities=capabilities
        )
        
        # Creative frameworks and templates
        self.creative_frameworks = {
            "storytelling": self._get_storytelling_framework(),
            "problem_solution": self._get_problem_solution_framework(),
            "benefit_focused": self._get_benefit_focused_framework(),
            "social_proof": self._get_social_proof_framework(),
            "urgency_based": self._get_urgency_based_framework()
        }
        
        # Content type specific generation strategies
        self.generation_strategies = {
            "ad": self._get_ad_generation_strategy(),
            "landing_page": self._get_landing_generation_strategy(),
            "blog_intro": self._get_blog_generation_strategy(),
            "email": self._get_email_generation_strategy(),
            "social_media": self._get_social_generation_strategy()
        }
        
        # Creative enhancement techniques
        self.enhancement_techniques = [
            "power_word_integration",
            "emotional_trigger_activation",
            "sensory_language_enhancement",
            "rhythm_and_flow_optimization",
            "curiosity_gap_creation"
        ]
        
        self.logger.info("Creative Generation Agent initialized with comprehensive creative frameworks")
    
    async def process_task(self, task: TaskRequest) -> Dict[str, Any]:
        """
        Execute creative content generation based on persona research and content strategy.
        
        This method orchestrates the complete creative process from strategic input
        to polished, ready-to-use marketing content.
        """
        self.logger.info(
            "Starting creative content generation",
            task_id=task.task_id,
            content_type=task.input_data.get("content_type"),
            topic=task.input_data.get("topic"),
            has_strategy=bool(task.input_data.get("content_strategy"))
        )
        
        # Extract input data
        persona_research = task.input_data.get("persona_research", {})
        content_strategy = task.input_data.get("content_strategy", {})
        content_type = task.input_data.get("content_type")
        topic = task.input_data.get("topic")
        word_count = task.input_data.get("word_count", 200)
        creativity_level = task.input_data.get("creativity_level", 0.7)
        brand_voice = task.input_data.get("brand_voice", {})
        
        if not content_strategy:
            raise ValueError("content_strategy is required for creative generation")
        if not content_type or not topic:
            raise ValueError("content_type and topic are required")
        
        # Build comprehensive creative brief
        creative_brief = self._build_creative_brief(
            persona_research, content_strategy, content_type, topic, 
            word_count, creativity_level, brand_voice
        )
        
        # Generate initial content using creative model
        primary_content = await self._generate_primary_content(
            creative_brief, content_type, creativity_level
        )
        
        # Create content variations for optimization
        content_variations = await self._generate_content_variations(
            primary_content, creative_brief, content_type
        )
        
        # Apply creative enhancements
        enhanced_content = await self._apply_creative_enhancements(
            primary_content, creative_brief, content_strategy
        )
        
        # Perform final optimization and formatting
        final_content = self._optimize_and_format_content(
            enhanced_content, content_type, word_count
        )
        
        # Generate creative metadata and insights
        creative_insights = self._analyze_creative_elements(
            final_content, creative_brief, content_strategy
        )
        
        # Store successful creative patterns for learning
        self._store_creative_patterns(final_content, creative_brief, content_type)
        
        result = {
            "generated_content": final_content,
            "content_variations": content_variations,
            "creative_confidence": self._calculate_creative_confidence(final_content, creative_brief),
            "creative_elements_analysis": creative_insights,
            "persona_alignment_score": self._calculate_persona_alignment(final_content, persona_research),
            "strategic_adherence": self._assess_strategic_adherence(final_content, content_strategy),
            "brand_voice_consistency": self._assess_brand_voice_consistency(final_content, brand_voice),
            "optimization_suggestions": self._generate_optimization_suggestions(final_content, content_type),
            "performance_predictions": self._predict_content_performance(final_content, persona_research),
            "tokens_used": 0,  # Will be updated with actual usage
            "api_calls": 0,    # Will be updated with actual calls
            "creative_metadata": {
                "content_type": content_type,
                "topic": topic,
                "word_count": len(final_content.split()),
                "creativity_level": creativity_level,
                "generation_timestamp": datetime.utcnow().isoformat(),
                "creative_approach": self._identify_creative_approach(final_content),
                "enhancement_techniques_used": self._identify_applied_techniques(final_content)
            }
        }
        
        self.logger.info(
            "Creative content generation completed",
            task_id=task.task_id,
            confidence=result["creative_confidence"],
            word_count=result["creative_metadata"]["word_count"],
            creative_approach=result["creative_metadata"]["creative_approach"]
        )
        
        return result
    
    async def _generate_primary_content(
        self, 
        creative_brief: str, 
        content_type: str, 
        creativity_level: float
    ) -> str:
        """
        Generate the primary content using the creative model with optimized parameters.
        """
        # Select model based on creativity requirements
        if creativity_level >= 0.8:
            model = settings.GROQ_MODEL_CREATIVE  # Use most creative model
            temperature = 0.8
        elif creativity_level >= 0.6:
            model = settings.GROQ_MODEL_SMART
            temperature = 0.7
        else:
            model = settings.GROQ_MODEL_SMART
            temperature = 0.5
        
        generation_response = await self.llm_service.generate_content(
            prompt=creative_brief,
            model=model,
            temperature=temperature,
            max_tokens=1200,
            task_type="creative"
        )
        
        return generation_response.content.strip()
    
    async def _generate_content_variations(
        self, 
        primary_content: str, 
        creative_brief: str, 
        content_type: str
    ) -> List[str]:
        """
        Generate alternative content variations for A/B testing and optimization.
        """
        variation_prompt = f"""
        Based on this primary content and creative brief, generate 3 alternative variations:
        
        PRIMARY CONTENT:
        {primary_content}
        
        CREATIVE BRIEF:
        {creative_brief}
        
        Create 3 distinct variations that:
        1. Variation A: More direct/logical approach
        2. Variation B: More emotional/story-driven approach  
        3. Variation C: More urgent/action-oriented approach
        
        Each variation should be the same length and format as the primary content.
        Separate each variation with "---VARIATION---"
        """
        
        try:
            variations_response = await self.llm_service.generate_content(
                prompt=variation_prompt,
                model=settings.GROQ_MODEL_CREATIVE,
                temperature=0.6,
                max_tokens=1500
            )
            
            # Parse variations
            variations = variations_response.content.split("---VARIATION---")
            cleaned_variations = [var.strip() for var in variations if var.strip()]
            
            return cleaned_variations[:3]  # Return first 3 variations
            
        except Exception as e:
            self.logger.warning(f"Failed to generate content variations: {e}")
            return []  # Return empty list if variation generation fails
    
    async def _apply_creative_enhancements(
        self, 
        content: str, 
        creative_brief: str, 
        content_strategy: Dict[str, Any]
    ) -> str:
        """
        Apply creative enhancement techniques to improve content impact and engagement.
        """
        enhancement_prompt = f"""
        Enhance this content using advanced copywriting techniques:
        
        ORIGINAL CONTENT:
        {content}
        
        CREATIVE BRIEF:
        {creative_brief}
        
        STRATEGIC CONTEXT:
        Core Messaging: {content_strategy.get('core_messaging', {})}
        Persuasion Techniques: {content_strategy.get('persuasion_techniques', {})}
        
        Apply these enhancements:
        1. Integrate power words for emotional impact
        2. Optimize rhythm and flow for readability
        3. Strengthen opening hook and closing CTA
        4. Add sensory language where appropriate
        5. Ensure perfect persona alignment
        
        Return the enhanced version maintaining the same structure and length.
        """
        
        try:
            enhancement_response = await self.llm_service.generate_content(
                prompt=enhancement_prompt,
                model=settings.GROQ_MODEL_SMART,
                temperature=0.4,
                max_tokens=800
            )
            
            enhanced_content = enhancement_response.content.strip()
            
            # Validate enhancement didn't break structure
            if len(enhanced_content) < len(content) * 0.7:
                self.logger.warning("Enhancement resulted in significantly shorter content, using original")
                return content
            
            return enhanced_content
            
        except Exception as e:
            self.logger.warning(f"Enhancement failed, using original content: {e}")
            return content
    
    def _optimize_and_format_content(
        self, 
        content: str, 
        content_type: str, 
        target_word_count: int
    ) -> str:
        """
        Apply final optimization and formatting based on content type requirements.
        """
        # Apply content-type specific formatting
        if content_type == "ad":
            return self._format_ad_content(content)
        elif content_type == "landing_page":
            return self._format_landing_page_content(content)
        elif content_type == "blog_intro":
            return self._format_blog_intro_content(content)
        elif content_type == "email":
            return self._format_email_content(content)
        elif content_type == "social_media":
            return self._format_social_media_content(content)
        else:
            return self._format_generic_content(content, target_word_count)
    
    def _build_creative_brief(
        self,
        persona_research: Dict[str, Any],
        content_strategy: Dict[str, Any],
        content_type: str,
        topic: str,
        word_count: int,
        creativity_level: float,
        brand_voice: Dict[str, Any]
    ) -> str:
        """
        Build a comprehensive creative brief that guides content generation.
        """
        # Extract key strategic elements
        core_messaging = content_strategy.get("core_messaging", {})
        content_structure = content_strategy.get("content_structure", {})
        persuasion_techniques = content_strategy.get("persuasion_techniques", {})
        messaging_optimization = content_strategy.get("messaging_optimization", {})
        cta_strategy = content_strategy.get("call_to_action", {})
        
        # Extract persona insights
        key_pain_points = persona_research.get("pain_points", [])[:3]
        key_goals = persona_research.get("goals", [])[:3]
        communication_prefs = persona_research.get("communication_preferences", {})
        
        creative_brief = f"""
        CREATIVE BRIEF FOR {content_type.upper()} CONTENT
        
        TOPIC: {topic}
        TARGET WORD COUNT: {word_count} words
        CREATIVITY LEVEL: {creativity_level}/1.0
        
        PERSONA INSIGHTS:
        Top Pain Points: {', '.join(key_pain_points)}
        Primary Goals: {', '.join(key_goals)}
        Communication Style: {communication_prefs.get('tone', 'professional')}
        Preferred Language: {communication_prefs.get('terminology', 'accessible')}
        
        STRATEGIC DIRECTION:
        Primary Value Proposition: {core_messaging.get('primary_value_proposition', 'Value-focused messaging')}
        Key Supporting Messages: {', '.join(core_messaging.get('supporting_messages', []))}
        Emotional Hooks: {', '.join(core_messaging.get('emotional_hooks', []))}
        
        CONTENT STRUCTURE STRATEGY:
        Opening Approach: {content_structure.get('opening_strategy', 'Problem-focused opening')}
        Body Framework: {content_structure.get('body_framework', 'Benefit-driven structure')}
        Closing Strategy: {content_structure.get('closing_strategy', 'Strong call-to-action')}
        
        PERSUASION STRATEGY:
        Primary Technique: {persuasion_techniques.get('primary_technique', 'Social proof')}
        Supporting Techniques: {', '.join(persuasion_techniques.get('supporting_techniques', []))}
        Trust Building: {persuasion_techniques.get('trust_building', 'Credibility-focused')}
        
        MESSAGING OPTIMIZATION:
        Tone Specification: {messaging_optimization.get('tone_specification', 'Professional and engaging')}
        Power Words: {', '.join(messaging_optimization.get('power_words', []))}
        Language Level: {messaging_optimization.get('language_level', 'Accessible')}
        
        CALL-TO-ACTION STRATEGY:
        Primary CTA: {cta_strategy.get('primary_cta', 'Take action now')}
        CTA Language: {cta_strategy.get('cta_language', 'Clear and direct')}
        Urgency Elements: {', '.join(cta_strategy.get('urgency_elements', []))}
        
        BRAND VOICE GUIDELINES:
        {self._format_brand_voice_guidelines(brand_voice)}
        
        CONTENT TYPE SPECIFICATIONS:
        {self.generation_strategies.get(content_type, self._get_generic_generation_strategy())}
        
        CREATIVE REQUIREMENTS:
        1. Lead with the strongest pain point or goal alignment
        2. Integrate the primary value proposition naturally
        3. Apply the specified persuasion techniques seamlessly
        4. Maintain the exact tone and language level specified
        5. Include a compelling call-to-action following the strategy
        6. Ensure every word serves the strategic objective
        7. Create content that feels authentic and personally relevant
        8. Optimize for the target word count and format requirements
        
        Generate compelling, conversion-focused content that perfectly balances strategic adherence 
        with creative excellence. The content should feel natural, engaging, and irresistibly 
        persuasive to the target persona.
        """
        
        return creative_brief
    
    def _format_brand_voice_guidelines(self, brand_voice: Dict[str, Any]) -> str:
        """Format brand voice guidelines for the creative brief."""
        if not brand_voice:
            return "Standard professional brand voice with authentic, trustworthy tone"
        
        guidelines = []
        if brand_voice.get("tone"):
            guidelines.append(f"Tone: {brand_voice['tone']}")
        if brand_voice.get("personality"):
            guidelines.append(f"Personality: {brand_voice['personality']}")
        if brand_voice.get("voice_attributes"):
            guidelines.append(f"Attributes: {', '.join(brand_voice['voice_attributes'])}")
        
        return ' | '.join(guidelines) if guidelines else "Authentic, professional brand voice"
    
    # Content formatting methods for different types
    def _format_ad_content(self, content: str) -> str:
        """Format content specifically for advertisement requirements."""
        # Ensure punchy, impactful formatting
        lines = content.strip().split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.strip():
                # Capitalize first letter of each sentence
                formatted_line = '. '.join(
                    sentence.strip().capitalize() for sentence in line.split('.')
                    if sentence.strip()
                )
                if formatted_line and not formatted_line.endswith('.'):
                    formatted_line += '.'
                formatted_lines.append(formatted_line)
        
        return '\n\n'.join(formatted_lines)
    
    def _format_landing_page_content(self, content: str) -> str:
        """Format content for landing page optimization."""
        # Structure for scannable, conversion-focused format
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        formatted_paragraphs = []
        for i, paragraph in enumerate(paragraphs):
            if i == 0:
                # First paragraph - headline style
                formatted_paragraphs.append(f"**{paragraph}**")
            elif "call to action" in paragraph.lower() or "click" in paragraph.lower():
                # CTA paragraph - emphasized
                formatted_paragraphs.append(f"*{paragraph}*")
            else:
                formatted_paragraphs.append(paragraph)
        
        return '\n\n'.join(formatted_paragraphs)
    
    def _format_blog_intro_content(self, content: str) -> str:
        """Format content as engaging blog introduction."""
        return content.strip()  # Blog intros typically need minimal formatting
    
    def _format_email_content(self, content: str) -> str:
        """Format content for email optimization."""
        # Ensure email-appropriate structure
        lines = content.strip().split('\n')
        formatted_lines = []
        
        for line in lines:
            if line.strip():
                # Keep lines concise for email readability
                if len(line) > 80:
                    # Break long lines at natural points
                    words = line.split()
                    current_line = []
                    
                    for word in words:
                        current_line.append(word)
                        if len(' '.join(current_line)) > 70:
                            formatted_lines.append(' '.join(current_line))
                            current_line = []
                    
                    if current_line:
                        formatted_lines.append(' '.join(current_line))
                else:
                    formatted_lines.append(line)
        
        return '\n\n'.join(formatted_lines)
    
    def _format_social_media_content(self, content: str) -> str:
        """Format content for social media platforms."""
        # Optimize for social media engagement
        content = content.strip()
        
        # Add hashtag-friendly formatting if needed
        if len(content) > 200:  # If too long for some platforms
            # Create shorter version
            sentences = content.split('. ')
            short_content = '. '.join(sentences[:2])
            if not short_content.endswith('.'):
                short_content += '.'
            return short_content
        
        return content
    
    def _format_generic_content(self, content: str, target_word_count: int) -> str:
        """Apply generic formatting optimizations."""
        words = content.strip().split()
        current_word_count = len(words)
        
        # Adjust length if significantly off target
        if current_word_count < target_word_count * 0.8:
            # Content is too short, flag for potential expansion
            pass
        elif current_word_count > target_word_count * 1.2:
            # Content is too long, trim carefully
            words = words[:target_word_count]
            content = ' '.join(words)
            # Ensure it ends properly
            if not content.endswith('.'):
                content = content.rsplit(' ', 1)[0] + '.'
        
        return content
    
    # Creative analysis and scoring methods
    def _calculate_creative_confidence(self, content: str, creative_brief: str) -> float:
        """Calculate confidence in the creative quality of generated content."""
        confidence_factors = []
        
        # Length appropriateness
        word_count = len(content.split())
        if 50 <= word_count <= 500:  # Reasonable range
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.6)
        
        # Structure quality (has beginning, middle, end)
        sentences = content.split('.')
        if len(sentences) >= 3:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.6)
        
        # Call-to-action presence
        cta_indicators = ["click", "get", "try", "start", "join", "buy", "learn", "discover", "find out"]
        if any(indicator in content.lower() for indicator in cta_indicators):
            confidence_factors.append(0.9)
        else:
            confidence_factors.append(0.5)
        
        # Emotional language presence
        emotion_words = ["amazing", "incredible", "transform", "breakthrough", "revolutionary", 
                        "exclusive", "limited", "powerful", "proven", "guaranteed"]
        emotion_count = sum(1 for word in emotion_words if word in content.lower())
        if emotion_count >= 2:
            confidence_factors.append(0.8)
        elif emotion_count >= 1:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        return sum(confidence_factors) / len(confidence_factors)
    
    def _calculate_persona_alignment(self, content: str, persona_research: Dict[str, Any]) -> float:
        """Calculate how well content aligns with persona characteristics."""
        alignment_score = 0.7  # Base score
        
        # Check pain point addressing
        pain_points = persona_research.get("pain_points", [])
        content_lower = content.lower()
        
        pain_points_addressed = 0
        for pain_point in pain_points[:3]:  # Check top 3 pain points
            pain_words = str(pain_point).lower().split()
            if any(word in content_lower for word in pain_words if len(word) > 3):
                pain_points_addressed += 1
        
        if pain_points_addressed > 0:
            alignment_score += min(0.2, pain_points_addressed / 3.0 * 0.3)
        
        # Check communication style alignment
        comm_prefs = persona_research.get("communication_preferences", {})
        preferred_tone = str(comm_prefs.get("tone", "")).lower()
        
        if preferred_tone:
            if "casual" in preferred_tone and any(word in content_lower for word in ["you", "your", "we", "us"]):
                alignment_score += 0.1
            elif "professional" in preferred_tone and not any(word in content_lower for word in ["hey", "hi", "wow"]):
                alignment_score += 0.1
        
        return min(alignment_score, 1.0)
    
    def _assess_strategic_adherence(self, content: str, content_strategy: Dict[str, Any]) -> float:
        """Assess how well content follows the strategic recommendations."""
        adherence_score = 0.6  # Base score
        content_lower = content.lower()
        
        # Check primary value proposition integration
        core_messaging = content_strategy.get("core_messaging", {})
        primary_value_prop = core_messaging.get("primary_value_proposition", "")
        
        if primary_value_prop:
            value_prop_words = primary_value_prop.lower().split()
            value_words_found = sum(1 for word in value_prop_words 
                                  if len(word) > 3 and word in content_lower)
            if value_words_found > 0:
                adherence_score += min(0.2, value_words_found / len(value_prop_words) * 0.3)
        
        # Check CTA strategy adherence
        cta_strategy = content_strategy.get("call_to_action", {})
        primary_cta = str(cta_strategy.get("primary_cta", "")).lower()
        
        if primary_cta and any(word in content_lower for word in primary_cta.split()):
            adherence_score += 0.15
        
        # Check persuasion technique application
        persuasion_techniques = content_strategy.get("persuasion_techniques", {})
        primary_technique = str(persuasion_techniques.get("primary_technique", "")).lower()
        
        if "social proof" in primary_technique and any(word in content_lower for word in ["customers", "users", "people", "others"]):
            adherence_score += 0.1
        elif "urgency" in primary_technique and any(word in content_lower for word in ["now", "today", "limited", "soon"]):
            adherence_score += 0.1
        elif "authority" in primary_technique and any(word in content_lower for word in ["expert", "proven", "research", "study"]):
            adherence_score += 0.1
        
        return min(adherence_score, 1.0)
    
    def _assess_brand_voice_consistency(self, content: str, brand_voice: Dict[str, Any]) -> float:
        """Assess brand voice consistency in the generated content."""
        if not brand_voice:
            return 0.8  # Default score when no brand voice specified
        
        consistency_score = 0.7  # Base score
        content_lower = content.lower()
        
        # Check tone consistency
        brand_tone = str(brand_voice.get("tone", "")).lower()
        if brand_tone:
            if "friendly" in brand_tone and any(word in content_lower for word in ["you", "your", "we", "us"]):
                consistency_score += 0.1
            elif "professional" in brand_tone and not any(word in content_lower for word in ["awesome", "cool", "hey"]):
                consistency_score += 0.1
        
        # Check personality attributes
        personality = brand_voice.get("personality", [])
        if personality:
            personality_indicators = 0
            for trait in personality:
                trait_lower = str(trait).lower()
                if "innovative" in trait_lower and any(word in content_lower for word in ["new", "advanced", "cutting-edge"]):
                    personality_indicators += 1
                elif "trustworthy" in trait_lower and any(word in content_lower for word in ["proven", "reliable", "guaranteed"]):
                    personality_indicators += 1
            
            if personality_indicators > 0:
                consistency_score += min(0.2, personality_indicators / len(personality) * 0.3)
        
        return min(consistency_score, 1.0)
    
    def _generate_optimization_suggestions(self, content: str, content_type: str) -> List[str]:
        """Generate specific suggestions for content optimization."""
        suggestions = []
        content_lower = content.lower()
        
        # General optimization suggestions
        if len(content.split()) < 50:
            suggestions.append("Consider expanding content to provide more value and context")
        
        if not any(word in content_lower for word in ["you", "your"]):
            suggestions.append("Add more personal pronouns to increase engagement")
        
        # Content-type specific suggestions
        if content_type == "ad":
            if not any(word in content_lower for word in ["now", "today", "limited"]):
                suggestions.append("Add urgency elements to drive immediate action")
            if content.count('!') == 0:
                suggestions.append("Consider adding exclamation points for emotional impact")
        
        elif content_type == "landing_page":
            if "benefits" not in content_lower and "benefit" not in content_lower:
                suggestions.append("Explicitly highlight key benefits for the target persona")
            if not any(word in content_lower for word in ["guarantee", "promise", "proven"]):
                suggestions.append("Add trust signals and risk-reduction elements")
        
        elif content_type == "email":
            if len(content.split('\n\n')[0].split()) > 30:
                suggestions.append("Shorten opening paragraph for better email engagement")
            if content.count('?') == 0:
                suggestions.append("Consider adding questions to increase interactivity")
        
        # Add A/B testing suggestions
        suggestions.append("Test different headlines for optimal performance")
        suggestions.append("Experiment with varying call-to-action language")
        
        return suggestions[:5]  # Return top 5 suggestions
    
    def _predict_content_performance(self, content: str, persona_research: Dict[str, Any]) -> Dict[str, Any]:
        """Predict content performance based on various quality indicators."""
        
        # Calculate engagement factors
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        
        # Reading level (simplified)
        avg_sentence_length = word_count / max(sentence_count, 1)
        readability_score = 1.0 - min(avg_sentence_length / 25.0, 0.4)  # Prefer shorter sentences
        
        # Emotional impact
        emotion_words = ["amazing", "incredible", "transform", "breakthrough", "exclusive", "powerful"]
        emotion_score = min(sum(1 for word in emotion_words if word in content.lower()) / 5.0, 1.0)
        
        # Call-to-action strength
        cta_indicators = ["click", "get", "try", "start", "join", "buy", "learn", "discover"]
        cta_score = min(sum(1 for word in cta_indicators if word in content.lower()) / 3.0, 1.0)
        
        # Overall performance prediction
        performance_score = (readability_score * 0.3 + emotion_score * 0.35 + cta_score * 0.35)
        
        return {
            "overall_performance_prediction": performance_score,
            "readability_score": readability_score,
            "emotional_impact_score": emotion_score,
            "cta_effectiveness_score": cta_score,
            "predicted_engagement_level": "high" if performance_score >= 0.8 else "medium" if performance_score >= 0.6 else "moderate",
            "optimization_priority": "low" if performance_score >= 0.8 else "medium" if performance_score >= 0.6 else "high",
            "performance_factors": {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "avg_sentence_length": round(avg_sentence_length, 1),
                "emotion_words_count": sum(1 for word in emotion_words if word in content.lower()),
                "cta_elements_count": sum(1 for word in cta_indicators if word in content.lower())
            }
        }
    
    def _analyze_creative_elements(self, content: str, creative_brief: str, content_strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze the creative elements used in the generated content."""
        
        analysis = {
            "creative_techniques_identified": [],
            "persuasion_elements": [],
            "emotional_triggers": [],
            "structural_elements": {},
            "language_characteristics": {}
        }
        
        content_lower = content.lower()
        
        # Identify creative techniques
        if any(word in content_lower for word in ["story", "imagine", "picture this"]):
            analysis["creative_techniques_identified"].append("storytelling")
        
        if any(word in content_lower for word in ["problem", "challenge", "struggle"]):
            analysis["creative_techniques_identified"].append("problem_identification")
        
        if any(word in content_lower for word in ["solution", "answer", "solves"]):
            analysis["creative_techniques_identified"].append("solution_presentation")
        
        # Identify persuasion elements
        if any(word in content_lower for word in ["customers", "users", "people love"]):
            analysis["persuasion_elements"].append("social_proof")
        
        if any(word in content_lower for word in ["proven", "research", "studies show"]):
            analysis["persuasion_elements"].append("authority")
        
        if any(word in content_lower for word in ["limited", "exclusive", "only"]):
            analysis["persuasion_elements"].append("scarcity")
        
        if any(word in content_lower for word in ["now", "today", "hurry"]):
            analysis["persuasion_elements"].append("urgency")
        
        # Identify emotional triggers
        positive_emotions = ["amazing", "incredible", "fantastic", "revolutionary", "breakthrough"]
        negative_emotions = ["worry", "fear", "struggle", "problem", "challenge"]
        
        for emotion in positive_emotions:
            if emotion in content_lower:
                analysis["emotional_triggers"].append(f"positive: {emotion}")
        
        for emotion in negative_emotions:
            if emotion in content_lower:
                analysis["emotional_triggers"].append(f"concern: {emotion}")
        
        # Analyze structure
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        analysis["structural_elements"] = {
            "opening_type": "hook" if sentences and any(word in sentences[0].lower() for word in ["imagine", "what if", "did you know"]) else "direct",
            "body_approach": "benefit_focused" if "benefit" in content_lower else "problem_solution",
            "closing_style": "cta_focused" if sentences and any(word in sentences[-1].lower() for word in ["click", "get", "start"]) else "informational"
        }
        
        # Analyze language characteristics
        analysis["language_characteristics"] = {
            "tone": "conversational" if content.count("you") > 2 else "professional",
            "complexity": "accessible" if sum(len(word) for word in content.split()) / len(content.split()) < 6 else "complex",
            "engagement_level": "high" if content.count("?") > 0 or content.count("!") > 1 else "moderate"
        }
        
        return analysis
    
    def _identify_creative_approach(self, content: str) -> str:
        """Identify the primary creative approach used in the content."""
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["story", "imagine", "picture", "once"]):
            return "storytelling"
        elif any(word in content_lower for word in ["problem", "challenge", "struggle"]) and any(word in content_lower for word in ["solution", "solve", "answer"]):
            return "problem_solution"
        elif any(word in content_lower for word in ["benefit", "advantage", "gain", "get"]):
            return "benefit_focused"
        elif any(word in content_lower for word in ["customers", "users", "people", "others"]):
            return "social_proof"
        elif any(word in content_lower for word in ["now", "today", "limited", "hurry"]):
            return "urgency_based"
        else:
            return "informational"
    
    def _identify_applied_techniques(self, content: str) -> List[str]:
        """Identify which enhancement techniques were successfully applied."""
        applied_techniques = []
        content_lower = content.lower()
        
        # Power word integration
        power_words = ["amazing", "incredible", "exclusive", "proven", "guaranteed", "revolutionary", "breakthrough"]
        if any(word in content_lower for word in power_words):
            applied_techniques.append("power_word_integration")
        
        # Emotional trigger activation
        if any(word in content_lower for word in ["feel", "experience", "imagine", "discover"]):
            applied_techniques.append("emotional_trigger_activation")
        
        # Sensory language
        sensory_words = ["see", "hear", "feel", "taste", "touch", "bright", "smooth", "crisp"]
        if any(word in content_lower for word in sensory_words):
            applied_techniques.append("sensory_language_enhancement")
        
        # Rhythm and flow (check for varied sentence lengths)
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        if len(sentences) > 1:
            sentence_lengths = [len(s.split()) for s in sentences]
            if max(sentence_lengths) - min(sentence_lengths) > 3:
                applied_techniques.append("rhythm_and_flow_optimization")
        
        # Curiosity gap creation
        if any(word in content_lower for word in ["discover", "secret", "revealed", "find out"]):
            applied_techniques.append("curiosity_gap_creation")
        
        return applied_techniques
    
    def _store_creative_patterns(self, content: str, creative_brief: str, content_type: str):
        """Store successful creative patterns for future learning and improvement."""
        # In a production system, this would store patterns in a database
        # For now, we'll store in agent memory
        
        pattern_key = f"successful_pattern_{content_type}_{hash(creative_brief[:100])}"
        pattern_data = {
            "content_sample": content[:200],  # Store first 200 chars
            "content_type": content_type,
            "success_indicators": {
                "word_count": len(content.split()),
                "creative_approach": self._identify_creative_approach(content),
                "techniques_used": self._identify_applied_techniques(content)
            },
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.add_memory(pattern_key, pattern_data, ttl=7200)  # Store for 2 hours
        
        self.logger.debug(
            "Stored creative pattern for future reference",
            pattern_key=pattern_key,
            content_type=content_type,
            approach=pattern_data["success_indicators"]["creative_approach"]
        )
    
    # Creative framework templates
    def _get_storytelling_framework(self) -> str:
        return """
        STORYTELLING FRAMEWORK:
        - Character introduction (relatable persona)
        - Challenge/conflict presentation
        - Journey and struggle elements
        - Solution discovery and transformation
        - Resolution and call-to-action
        """
    
    def _get_problem_solution_framework(self) -> str:
        return """
        PROBLEM-SOLUTION FRAMEWORK:
        - Problem identification and agitation
        - Consequences of inaction
        - Solution introduction
        - Proof of effectiveness
        - Implementation call-to-action
        """
    
    def _get_benefit_focused_framework(self) -> str:
        return """
        BENEFIT-FOCUSED FRAMEWORK:
        - Primary benefit headline
        - Supporting benefit enumeration
        - Feature-to-benefit translation
        - Value demonstration
        - Benefit realization call-to-action
        """
    
    def _get_social_proof_framework(self) -> str:
        return """
        SOCIAL PROOF FRAMEWORK:
        - Community/user base introduction
        - Success stories and testimonials
        - Statistical proof points
        - Expert endorsements
        - Join-the-community call-to-action
        """
    
    def _get_urgency_based_framework(self) -> str:
        return """
        URGENCY-BASED FRAMEWORK:
        - Opportunity introduction
        - Scarcity or time limitation
        - Consequences of delay
        - Immediate action benefits
        - Urgent call-to-action
        """
    
    # Content type generation strategies
    def _get_ad_generation_strategy(self) -> str:
        return """
        ADVERTISEMENT GENERATION STRATEGY:
        - Hook: Attention-grabbing opening (3-5 words)
        - Problem/Desire: Quick pain point or aspiration
        - Solution: Clear value proposition
        - Proof: Brief credibility element
        - Call-to-Action: Direct, action-oriented
        Target: 25-75 words maximum
        """
    
    def _get_landing_generation_strategy(self) -> str:
        return """
        LANDING PAGE GENERATION STRATEGY:
        - Headline: Primary value proposition
        - Subheadline: Clarification and benefits
        - Body: 3-5 key benefits with proof
        - Social Proof: Testimonial or statistic
        - Call-to-Action: Prominent and clear
        Target: 150-400 words
        """
    
    def _get_blog_generation_strategy(self) -> str:
        return """
        BLOG INTRODUCTION GENERATION STRATEGY:
        - Hook: Question, statistic, or surprising fact
        - Problem: Reader's pain point or interest
        - Promise: What the article will deliver
        - Preview: Brief outline of key points
        - Transition: Smooth entry to main content
        Target: 100-200 words
        """
    
    def _get_email_generation_strategy(self) -> str:
        return """
        EMAIL GENERATION STRATEGY:
        - Subject Preview: Hook for opening
        - Personal Greeting: Direct, relevant opener
        - Value Delivery: Core message or offer
        - Benefit Explanation: Why it matters
        - Clear Call-to-Action: Single, focused action
        Target: 75-250 words
        """
    
    def _get_social_generation_strategy(self) -> str:
        return """
        SOCIAL MEDIA GENERATION STRATEGY:
        - Attention Grabber: Visual or emotional hook
        - Core Message: Single key point
        - Engagement Element: Question or interaction
        - Hashtag Strategy: Relevant community tags
        - Call-to-Action: Simple, shareable action
        Target: Platform-specific length optimization
        """
    
    def _get_generic_generation_strategy(self) -> str:
        return """
        GENERIC CONTENT GENERATION STRATEGY:
        - Opening: Relevant hook or introduction
        - Body: Main message with supporting points
        - Benefits: Clear value proposition
        - Proof: Credibility or social proof elements
        - Closing: Strong call-to-action
        Target: Flexible based on requirements
        """
