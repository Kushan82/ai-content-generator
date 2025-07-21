from pydantic import BaseModel, Field, validator
from typing import Dict, Any, List, Optional, Literal
from enum import Enum
from datetime import datetime
import uuid

class PersonaType(str, Enum):
    """Supported persona types for content generation"""
    YOUNG_PARENT = "young_parent"
    STARTUP_CTO = "startup_cto"
    ENTERPRISE_EXEC = "enterprise_exec"
    SMALL_BUSINESS_OWNER = "small_business_owner"
    MARKETING_MANAGER = "marketing_manager"

class ContentType(str, Enum):
    """Supported content types for generation"""
    ADVERTISEMENT = "ad"
    LANDING_PAGE = "landing_page"
    BLOG_INTRO = "blog_intro"
    EMAIL_CAMPAIGN = "email"
    SOCIAL_MEDIA = "social_media"
    PRODUCT_DESCRIPTION = "product_description"

class Persona(BaseModel):
    """
    Comprehensive persona model with detailed demographic and psychographic data
    """
    id: str = Field(..., description="Unique persona identifier")
    name: str = Field(..., description="Human-readable persona name")
    type: PersonaType
    
    # Demographic Information
    demographics: Dict[str, str] = Field(
        ..., 
        description="Age, income, location, education, family status",
        example={
            "age_range": "28-35",
            "income": "$50k-75k", 
            "location": "Suburban",
            "education": "College graduate",
            "family_status": "Married with children"
        }
    )
    
    # Psychographic Profile
    pain_points: List[str] = Field(
        ..., 
        description="Key challenges and problems this persona faces",
        min_items=3,
        max_items=10
    )
    
    goals: List[str] = Field(
        ..., 
        description="Primary objectives and aspirations",
        min_items=3,
        max_items=8
    )
    
    # Communication Preferences
    communication_preferences: Dict[str, str] = Field(
        ..., 
        description="Preferred tone, formality, terminology",
        example={
            "tone": "warm, empathetic",
            "formality": "casual but respectful",
            "terminology": "simple, avoid jargon",
            "urgency_preference": "moderate"
        }
    )
    
    # Content Preferences
    content_preferences: Dict[str, str] = Field(
        ..., 
        description="Preferred content length, format, style",
        example={
            "length": "concise, scannable",
            "format": "bullet points, short paragraphs",
            "visual_style": "clean, family-friendly",
            "emotional_appeal": "safety, security, family-focused"
        }
    )
    
    # Digital Behavior
    digital_behavior: Dict[str, Any] = Field(
        default_factory=dict,
        description="Online behavior patterns, platform preferences"
    )
    
    @validator('pain_points')
    def validate_pain_points(cls, v):
        if len(v) < 3:
            raise ValueError('Persona must have at least 3 pain points')
        return v

class ContentRequest(BaseModel):
    """
    Content generation request with comprehensive parameters
    """
    # Required Fields
    persona_id: str = Field(..., description="Target persona identifier")
    content_type: ContentType = Field(..., description="Type of content to generate")
    topic: str = Field(
        ..., 
        min_length=5, 
        max_length=200,
        description="Main topic, product, or service"
    )
    
    # Optional Customization
    additional_context: Optional[str] = Field(
        None, 
        max_length=1000,
        description="Additional requirements, constraints, or context"
    )
    
    word_count: int = Field(
        default=200, 
        ge=50, 
        le=1000,
        description="Target word count for generated content"
    )
    
    # Advanced Parameters
    creativity_level: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Creativity level (0=conservative, 1=highly creative)"
    )
    
    urgency_level: Literal["low", "medium", "high"] = Field(
        default="medium",
        description="Urgency level for call-to-action"
    )
    
    include_cta: bool = Field(
        default=True,
        description="Whether to include call-to-action"
    )
    
    target_emotion: Optional[str] = Field(
        None,
        description="Primary emotion to evoke (excitement, trust, urgency, etc.)"
    )
    
    # Request Metadata
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    @validator('topic')
    def validate_topic(cls, v):
        if len(v.strip()) < 5:
            raise ValueError('Topic must be at least 5 characters long')
        return v.strip()

class QualityMetrics(BaseModel):
    """
    Quality assessment metrics for generated content
    """
    overall_score: float = Field(ge=0.0, le=1.0, description="Overall quality score")
    clarity_score: float = Field(ge=0.0, le=1.0, description="Content clarity rating")
    engagement_score: float = Field(ge=0.0, le=1.0, description="Engagement potential")
    persuasion_score: float = Field(ge=0.0, le=1.0, description="Persuasive effectiveness")
    persona_alignment: float = Field(ge=0.0, le=1.0, description="Persona alignment score")
    
    # Detailed assessments
    strengths: List[str] = Field(default_factory=list, description="Content strengths")
    weaknesses: List[str] = Field(default_factory=list, description="Areas for improvement")
    recommendations: List[str] = Field(default_factory=list, description="Improvement suggestions")

class ContentResponse(BaseModel):
    """
    Final content generation response with multi-agent contributions
    """
    # Generated Content
    final_content: str = Field(..., description="Final optimized content")
    content_variants: List[str] = Field(
        default_factory=list,
        description="Alternative content versions"
    )
    
    # Request Information
    request: ContentRequest
    persona_used: Persona
    
    # Quality Assessment
    quality_metrics: QualityMetrics = Field(..., description="Quality assessment results")
    
    # Generation Metadata
    generation_metadata: Dict[str, Any] = Field(
        default_factory=lambda: {
            "generation_id": str(uuid.uuid4()),
            "model_used": "groq-llama3.1-70b",
            "total_tokens": 0,
            "generation_time": 0.0,
            "agents_involved": 0,
            "workflow_version": "2.0"
        }
    )
    
    timestamp: datetime = Field(default_factory=datetime.utcnow)
