"""Input validation utilities"""
from typing import Dict, Any, List
from pydantic import ValidationError
from app.core.exceptions import MultiAgentError

class ContentRequestValidator:
    """Validates content generation requests"""
    
    @staticmethod
    def validate_content_type(content_type: str) -> bool:
        allowed_types = ['ad', 'landing_page', 'blog_intro', 'email', 'social_media']
        if content_type not in allowed_types:
            raise MultiAgentError(
                f"Invalid content type: {content_type}",
                error_code="INVALID_CONTENT_TYPE"
            )
        return True
    
    @staticmethod
    def validate_word_count(word_count: int, content_type: str) -> bool:
        limits = {
            'ad': (25, 100),
            'landing_page': (150, 500),
            'email': (75, 300),
            'social_media': (20, 280)
        }
        
        if content_type in limits:
            min_words, max_words = limits[content_type]
            if not (min_words <= word_count <= max_words):
                raise MultiAgentError(
                    f"Word count {word_count} outside range {min_words}-{max_words} for {content_type}",
                    error_code="INVALID_WORD_COUNT"
                )
        return True
