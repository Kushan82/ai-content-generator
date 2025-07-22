"""Persona management service"""
from typing import List, Optional, Dict, Any
from app.models.content_schemas import Persona
from app.core.exceptions import PersonaNotFoundError

class PersonaService:
    """Service for managing persona data and operations"""
    
    def __init__(self):
        self._personas = self._load_default_personas()
    
    async def get_persona(self, persona_id: str) -> Persona:
        """Get persona by ID with error handling"""
        persona = next((p for p in self._personas if p.id == persona_id), None)
        if not persona:
            raise PersonaNotFoundError(
                f"Persona '{persona_id}' not found",
                error_code="PERSONA_NOT_FOUND"
            )
        return persona
    
    async def list_personas(self) -> List[Persona]:
        """List all available personas"""
        return self._personas
    
    async def validate_persona_compatibility(self, persona_id: str, content_type: str) -> bool:
        """Validate if persona is compatible with content type"""
        persona = await self.get_persona(persona_id)
        # Add compatibility logic here
        return True
    
    def _load_default_personas(self) -> List[Persona]:
        """Load default persona profiles"""
        return [
            Persona(
                id="young_parent",
                name="Young Parent",
                type="young_parent",
                demographics={
                    "age_range": "28-35",
                    "income": "$50k-75k",
                    "location": "Suburban"
                },
                pain_points=[
                    "Limited time for research",
                    "Budget constraints",
                    "Safety concerns for children"
                ],
                goals=[
                    "Provide best for children",
                    "Save time and money",
                    "Maintain family health"
                ],
                communication_preferences={
                    "tone": "warm, empathetic",
                    "formality": "casual but respectful"
                },
                content_preferences={
                    "length": "concise, scannable",
                    "format": "bullet points, short paragraphs"
                }
            )
            # Add more personas...
        ]
