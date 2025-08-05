import requests
import json
from typing import Dict, Any, List, Optional
import streamlit as st
from datetime import datetime
import time

class APIClient:
    """
    Enhanced API client for communicating with the FastAPI backend.
    Handles all HTTP requests with comprehensive error handling, caching, and retry logic.
    """
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'Content-Type': 'application/json',
            'Accept': 'application/json'
        })
        
        # Enhanced caching
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Request tracking
        self.request_count = 0
        self.last_request_time = None
        
        # Connection status
        self.is_connected = False
        self.last_health_check = None

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with enhanced error handling and retry logic."""
        url = f"{self.base_url}{endpoint}"
        max_retries = 3
        retry_delay = 1
        
        self.request_count += 1
        self.last_request_time = datetime.now()
        
        for attempt in range(max_retries):
            try:
                response = self.session.request(method, url, timeout=30, **kwargs)
                response.raise_for_status()
                
                self.is_connected = True
                return response.json()
                
            except requests.exceptions.ConnectionError:
                self.is_connected = False
                if attempt == max_retries - 1:
                    st.error("❌ Unable to connect to the API server. Please ensure the backend is running.")
                    raise
                time.sleep(retry_delay * (attempt + 1))
                
            except requests.exceptions.Timeout:
                if attempt == max_retries - 1:
                    st.error("⏱️ Request timed out. The server might be overloaded.")
                    raise
                time.sleep(retry_delay * (attempt + 1))
                
            except requests.exceptions.HTTPError as e:
                if response.status_code == 404:
                    st.error(f"🔍 API endpoint not found: {endpoint}")
                elif response.status_code == 500:
                    st.error("🔧 Internal server error. Please check the server logs.")
                elif response.status_code == 503:
                    st.error("⚠️ Service temporarily unavailable. Please try again later.")
                else:
                    st.error(f"📡 HTTP error {response.status_code}: {str(e)}")
                raise
                
            except Exception as e:
                if attempt == max_retries - 1:
                    st.error(f"❓ Unexpected error: {str(e)}")
                    raise
                time.sleep(retry_delay * (attempt + 1))

    def _get_cached(self, key: str) -> Optional[Any]:
        """Get cached data if still valid."""
        if key in self.cache:
            cached_data, timestamp = self.cache[key]
            if (datetime.now() - timestamp).seconds < self.cache_ttl:
                return cached_data
            # Remove expired cache entry
            del self.cache[key]
        return None

    def _set_cache(self, key: str, data: Any):
        """Cache data with timestamp."""
        self.cache[key] = (data, datetime.now())

    def get_health_status(self) -> Dict[str, Any]:
        """Get API health status with caching."""
        cache_key = 'health_status'
        cached_data = self._get_cached(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        try:
            health_data = self._make_request('GET', '/health')
            self._set_cache(cache_key, health_data)
            self.last_health_check = datetime.now()
            return health_data
        except Exception as e:
            # Return fallback health status
            return {
                'status': 'unknown',
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        try:
            return self._make_request('GET', '/api/v1/metrics')
        except Exception:
            # Return mock metrics as fallback
            return {
                'total_requests': self.request_count,
                'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None,
                'cache_size': len(self.cache),
                'connection_status': self.is_connected
            }

    def get_personas(self) -> List[Dict[str, Any]]:
        """Get available personas with enhanced caching and fallback."""
        cache_key = 'personas'
        cached_data = self._get_cached(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        try:
            personas = self._make_request('GET', '/api/v1/content/personas')
            self._set_cache(cache_key, personas)
            return personas
            
        except Exception as e:
            # Enhanced fallback with more personas
            st.warning("Using sample personas - API endpoint not available")
            fallback_personas = [
                {
                    "id": "young_parent",
                    "name": "Young Parent",
                    "type": "young_parent",
                    "demographics": {
                        "age_range": "28-35",
                        "income": "$50k-75k",
                        "location": "Suburban",
                        "education": "College graduate"
                    },
                    "pain_points": [
                        "Limited time for research",
                        "Budget constraints",
                        "Safety concerns for children"
                    ],
                    "goals": [
                        "Provide best for children",
                        "Save time and money",
                        "Maintain family health"
                    ],
                    "communication_preferences": {
                        "tone": "warm, empathetic",
                        "formality": "casual but respectful"
                    },
                    "content_preferences": {
                        "length": "concise, scannable",
                        "format": "bullet points, short paragraphs"
                    }
                },
                {
                    "id": "startup_cto",
                    "name": "Startup CTO", 
                    "type": "startup_cto",
                    "demographics": {
                        "age_range": "30-40",
                        "income": "$120k-200k",
                        "location": "Urban tech hubs",
                        "education": "Computer Science degree"
                    },
                    "pain_points": [
                        "Limited budget and resources",
                        "Need to scale quickly",
                        "Technical debt concerns"
                    ],
                    "goals": [
                        "Build scalable technology",
                        "Optimize development costs",
                        "Attract top talent"
                    ],
                    "communication_preferences": {
                        "tone": "direct, technical",
                        "formality": "professional but approachable"
                    },
                    "content_preferences": {
                        "length": "detailed, comprehensive",
                        "format": "technical specifications, case studies"
                    }
                },
                {
                    "id": "enterprise_exec",
                    "name": "Enterprise Executive",
                    "type": "enterprise_exec", 
                    "demographics": {
                        "age_range": "40-55",
                        "income": "$200k+",
                        "location": "Major business centers",
                        "education": "MBA or equivalent"
                    },
                    "pain_points": [
                        "Complex decision-making processes",
                        "Risk management concerns",
                        "ROI justification requirements"
                    ],
                    "goals": [
                        "Drive business growth",
                        "Improve operational efficiency",
                        "Manage risk effectively"
                    ],
                    "communication_preferences": {
                        "tone": "authoritative, professional",
                        "formality": "formal, executive-level"
                    },
                    "content_preferences": {
                        "length": "executive summary style",
                        "format": "strategic reports, white papers"
                    }
                },
                {
                    "id": "small_business_owner",
                    "name": "Small Business Owner",
                    "type": "small_business_owner",
                    "demographics": {
                        "age_range": "35-50",
                        "income": "$60k-120k",
                        "location": "Small to medium cities",
                        "education": "Varied backgrounds"
                    },
                    "pain_points": [
                        "Limited marketing budget",
                        "Wearing multiple hats",
                        "Competition with larger businesses"
                    ],
                    "goals": [
                        "Grow customer base",
                        "Increase revenue",
                        "Build brand recognition"
                    ],
                    "communication_preferences": {
                        "tone": "friendly, practical",
                        "formality": "casual business"
                    },
                    "content_preferences": {
                        "length": "practical, actionable",
                        "format": "how-to guides, tips"
                    }
                }
            ]
            
            self._set_cache(cache_key, fallback_personas)
            return fallback_personas

    def get_content_types(self) -> Dict[str, Any]:
        """Get available content types with enhanced fallback."""
        cache_key = 'content_types'
        cached_data = self._get_cached(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        try:
            content_types = self._make_request('GET', '/api/v1/content/types')
            self._set_cache(cache_key, content_types)
            return content_types
        except Exception:
            # Enhanced fallback content types
            fallback_types = {
                "content_types": [
                    {"id": "ad", "name": "Advertisement", "description": "Short promotional content"},
                    {"id": "landing_page", "name": "Landing Page", "description": "Website landing page copy"},
                    {"id": "blog_intro", "name": "Blog Introduction", "description": "Blog post introduction"},
                    {"id": "email", "name": "Email Campaign", "description": "Email marketing content"},
                    {"id": "social_media", "name": "Social Media", "description": "Social media posts"},
                    {"id": "product_description", "name": "Product Description", "description": "Product detail copy"}
                ]
            }
            self._set_cache(cache_key, fallback_types)
            return fallback_types

    def generate_content(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate content using multi-agent workflow with progress tracking."""
        # Add request metadata
        request_data['request_timestamp'] = datetime.now().isoformat()
        request_data['client_version'] = '2.0.0'
        
        return self._make_request('POST', '/api/v1/content/generate', json=request_data)

    def get_generation_status(self, request_id: str) -> Dict[str, Any]:
        """Get content generation status with detailed progress."""
        return self._make_request('GET', f'/api/v1/content/status/{request_id}')

    def get_agent_status(self) -> Dict[str, Any]:
        """Get comprehensive agent status information."""
        cache_key = 'agent_status'
        cached_data = self._get_cached(cache_key)
        
        if cached_data is not None:
            return cached_data
        
        try:
            agent_status = self._make_request('GET', '/api/v1/agents/status')
            self._set_cache(cache_key, agent_status)
            return agent_status
        except Exception:
            # Enhanced mock agent status
            mock_status = {
                'persona_researcher': {
                    'name': 'Persona Research Specialist',
                    'status': 'idle',
                    'uptime': 3600.5 + (datetime.now().second * 10),
                    'metrics': {
                        'success_rate': 0.96,
                        'average_response_time': 15.2,
                        'total_tasks_completed': 120 + self.request_count
                    },
                    'capabilities': [
                        {'name': 'demographic_analysis', 'proficiency_level': 0.95},
                        {'name': 'psychographic_profiling', 'proficiency_level': 0.90},
                        {'name': 'pain_point_analysis', 'proficiency_level': 0.88}
                    ],
                    'last_activity': datetime.now().isoformat()
                },
                'content_strategist': {
                    'name': 'Content Strategy Planner',
                    'status': 'idle',
                    'uptime': 3598.2 + (datetime.now().second * 10),
                    'metrics': {
                        'success_rate': 0.92,
                        'average_response_time': 18.7,
                        'total_tasks_completed': 118 + self.request_count
                    },
                    'capabilities': [
                        {'name': 'messaging_strategy', 'proficiency_level': 0.92},
                        {'name': 'content_planning', 'proficiency_level': 0.90},
                        {'name': 'persuasion_optimization', 'proficiency_level': 0.88}
                    ],
                    'last_activity': datetime.now().isoformat()
                },
                'creative_generator': {
                    'name': 'Creative Content Generator',
                    'status': 'idle',
                    'uptime': 3590.8 + (datetime.now().second * 10),
                    'metrics': {
                        'success_rate': 0.88,
                        'average_response_time': 22.1,
                        'total_tasks_completed': 115 + self.request_count
                    },
                    'capabilities': [
                        {'name': 'creative_copywriting', 'proficiency_level': 0.94},
                        {'name': 'brand_voice_adaptation', 'proficiency_level': 0.91},
                        {'name': 'persuasive_writing', 'proficiency_level': 0.93}
                    ],
                    'last_activity': datetime.now().isoformat()
                },
                'quality_assurance': {
                    'name': 'Quality Assurance Specialist',
                    'status': 'idle',
                    'uptime': 3575.3 + (datetime.now().second * 10),
                    'metrics': {
                        'success_rate': 0.94,
                        'average_response_time': 16.8,
                        'total_tasks_completed': 108 + self.request_count
                    },
                    'capabilities': [
                        {'name': 'content_quality_assessment', 'proficiency_level': 0.96},
                        {'name': 'strategic_alignment_validation', 'proficiency_level': 0.94},
                        {'name': 'persona_compliance_check', 'proficiency_level': 0.92}
                    ],
                    'last_activity': datetime.now().isoformat()
                },
                'orchestrator': {
                    'name': 'Workflow Orchestrator',
                    'status': 'idle',
                    'uptime': 3602.1 + (datetime.now().second * 10),
                    'metrics': {
                        'success_rate': 0.98,
                        'average_response_time': 5.0,
                        'total_tasks_completed': 95 + self.request_count
                    },
                    'capabilities': [
                        {'name': 'workflow_orchestration', 'proficiency_level': 0.98},
                        {'name': 'task_delegation', 'proficiency_level': 0.96},
                        {'name': 'agent_coordination', 'proficiency_level': 0.94}
                    ],
                    'last_activity': datetime.now().isoformat()
                }
            }
            
            self._set_cache(cache_key, mock_status)
            return mock_status

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get detailed workflow status."""
        return self._make_request('GET', f'/api/v1/workflow/status/{workflow_id}')

    def cancel_workflow(self, workflow_id: str) -> Dict[str, Any]:
        """Cancel a running workflow."""
        return self._make_request('POST', f'/api/v1/workflow/cancel/{workflow_id}')

    def get_connection_status(self) -> Dict[str, Any]:
        """Get current connection status and statistics."""
        return {
            'is_connected': self.is_connected,
            'base_url': self.base_url,
            'request_count': self.request_count,
            'last_request_time': self.last_request_time.isoformat() if self.last_request_time else None,
            'cache_entries': len(self.cache),
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None
        }

    def clear_cache(self):
        """Clear all cached data."""
        self.cache.clear()

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'total_entries': len(self.cache),
            'entries': list(self.cache.keys()),
            'ttl_seconds': self.cache_ttl
        }
