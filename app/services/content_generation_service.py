"""High-level content generation service"""
from typing import Dict, Any, Optional
from app.models.content_schemas import MultiAgentContentRequest, ContentGenerationResponse
from app.services.workflow_service import WorkflowService
from app.services.persona_service import PersonaService
from app.core.validators import ContentRequestValidator

class ContentGenerationService:
    """High-level service for content generation orchestration"""
    
    def __init__(self, workflow_service: WorkflowService, persona_service: PersonaService):
        self.workflow_service = workflow_service
        self.persona_service = persona_service
        self.validator = ContentRequestValidator()
    
    async def generate_content(self, request: MultiAgentContentRequest) -> ContentGenerationResponse:
        """Generate content with full validation and error handling"""
        # Validate request
        await self._validate_request(request)
        
        # Execute workflow
        workflow_result = await self.workflow_service.execute_content_workflow(request)
        
        # Transform and return result
        return self._transform_result(workflow_result, request)
    
    async def _validate_request(self, request: MultiAgentContentRequest):
        """Comprehensive request validation"""
        # Validate content type
        self.validator.validate_content_type(request.content_type)
        
        # Validate word count
        self.validator.validate_word_count(request.word_count, request.content_type)
        
        # Validate persona exists
        await self.persona_service.get_persona(request.persona_id)
        
        # Validate persona-content compatibility
        await self.persona_service.validate_persona_compatibility(
            request.persona_id, 
            request.content_type
        )
