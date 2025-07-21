import requests
import json
from typing import Dict, Any, List, Optional
import streamlit as st
from datetime import datetime

class APIClient:
    """
    API client for communicating with the FastAPI backend.
    Handles all HTTP requests with error handling and caching.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Cache for reducing API calls
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with error handling."""
        url = f"{self.base_url}{endpoint}"
        
        try:
            response = self.session.request(method, url, timeout=30, **kwargs)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.ConnectionError:
            st.error("❌ Unable to connect to the API server. Please ensure the backend is running.")
            raise
        
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timed out. The server might be overloaded.")
            raise
        
        except requests.exceptions.HTTPError as e:
            if response.status_code == 404:
                st.error(f"🔍 API endpoint not found: {endpoint}")
            elif response.status_code == 500:
                st.error("🔧 Internal server error. Please check the server logs.")
            else:
                st.error(f"📡 HTTP error {response.status_code}: {str(e)}")
            raise
        
        except Exception as e:
            st.error(f"❓ Unexpected error: {str(e)}")
            raise
    
    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached data if still valid."""
        if key in self.cache:
            cached_data, timestamp = self.cache[key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return cached_data
        return None
    
    def _set_cache(self, key: str, data: Any):
        """Cache data with timestamp."""
        self.cache[key] = (data, datetime.now())
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get API health status."""
        return self._make_request('GET', '/health')
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get system metrics."""
        return self._make_request('GET', '/metrics')
    
    def get_personas(self) -> List[Dict[str, Any]]:
        """Get available personas with caching."""
        cache_key = 'personas'
        cached_data = self._get_cached(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        personas = self._make_request('GET', '/api/v1/content/personas')
        self._set_cache(cache_key, personas)
        return personas
    
    def get_content_types(self) -> Dict[str, Any]:
        """Get available content types with caching."""
        cache_key = 'content_types'
        cached_data = self._get_cached(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        content_types = self._make_request('GET', '/api/v1/content/types')
        self._set_cache(cache_key, content_types)
        return content_types
    
    def generate_content(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content using multi-agent workflow."""
        return self._make_request('POST', '/api/v1/content/generate', json=request_data)
    
    def get_generation_status(self, request_id: str) -> Dict[str, Any]:
        """Get content generation status."""
        return self._make_request('GET', f'/api/v1/content/status/{request_id}')
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get agent status information."""
        # Mock agent status (in production, this would be a real API call)
        return {
            'persona_researcher': {
                'name': 'Persona Research Specialist',
                'status': 'idle',
                'uptime': 3600.5,
                'metrics': {
                    'success_rate': 0.96,
                    'average_response_time': 15.2,
                    'total_tasks_completed': 120
                },
                'capabilities': [
                    {'name': 'demographic_analysis', 'proficiency_level': 0.95},
                    {'name': 'psychographic_profiling', 'proficiency_level': 0.90},
                    {'name': 'pain_point_analysis', 'proficiency_level': 0.88}
                ]
            },
            'content_strategist': {
                'name': 'Content Strategy Planner',
                'status': 'idle',
                'uptime': 3598.2,
                'metrics': {
                    'success_rate': 0.92,
                    'average_response_time': 18.7,
                    'total_tasks_completed': 118
                },
                'capabilities': [
                    {'name': 'messaging_strategy', 'proficiency_level': 0.92},
                    {'name': 'content_planning', 'proficiency_level': 0.90},
                    {'name': 'persuasion_optimization', 'proficiency_level': 0.88}
                ]
            },
            'creative_generator': {
                'name': 'Creative Content Generator',
                'status': 'idle',
                'uptime': 3590.8,
                'metrics': {
                    'success_rate': 0.88,
                    'average_response_time': 22.1,
                    'total_tasks_completed': 115
                },
                'capabilities': [
                    {'name': 'creative_copywriting', 'proficiency_level': 0.94},
                    {'name': 'brand_voice_adaptation', 'proficiency_level': 0.91},
                    {'name': 'persuasive_writing', 'proficiency_level': 0.93}
                ]
            },
            'quality_assurance': {
                'name': 'Quality Assurance Specialist',
                'status': 'idle',
                'uptime': 3575.3,
                'metrics': {
                    'success_rate': 0.94,
                    'average_response_time': 16.8,
                    'total_tasks_completed': 108
                },
                'capabilities': [
                    {'name': 'content_quality_assessment', 'proficiency_level': 0.96},
                    {'name': 'strategic_alignment_validation', 'proficiency_level': 0.94},
                    {'name': 'persona_compliance_check', 'proficiency_level': 0.92}
                ]
            },
            'orchestrator': {
                'name': 'Workflow Orchestrator',
                'status': 'idle',
                'uptime': 3602.1,
                'metrics': {
                    'success_rate': 0.98,
                    'average_response_time': 5.0,
                    'total_tasks_completed': 95
                },
                'capabilities': [
                    {'name': 'workflow_orchestration', 'proficiency_level': 0.98},
                    {'name': 'task_delegation', 'proficiency_level': 0.96},
                    {'name': 'agent_coordination', 'proficiency_level': 0.94}
                ]
            }
        }
    
    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get workflow status."""
        return self._make_request('GET', f'/api/v1/workflow/status/{workflow_id}')
    
    def cancel_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Cancel a running workflow."""
        return self._make_request('POST', f'/api/v1/workflow/cancel/{workflow_id}')
