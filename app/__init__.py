"""
Multi-Agent AI Content Generator

A production-ready content generation platform using collaborative AI agents.
Built with LangGraph, CrewAI, FastAPI, and Streamlit.
"""

__version__ = "2.0.0"
__author__ = "AI Content Generator Team"
__description__ = "Multi-Agent AI Content Generation Platform"

# Package-level imports for convenience
from app.core.config import settings
from app.core.logging import logger

__all__ = ["settings", "logger", "__version__"]
