import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

from groq import AsyncGroq
from pydantic import BaseModel, Field
import structlog

from app.core.config import settings
from app.core.logging import logger

class LLMRequest(BaseModel):
    """
    Standardized request format for LLM operations.
    Encapsulates all parameters needed for content generation.
    """
    prompt: str = Field(..., min_length=1, description="The input prompt for the LLM")
    model: str = Field(default=settings.GROQ_MODEL_SMART, description="Model to use for generation")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Creativity level")
    max_tokens: int = Field(default=1000, ge=1, le=4000, description="Maximum tokens to generate")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    
    # Context and metadata
    system_prompt: Optional[str] = Field(None, description="System instruction for the model")
    context: Dict[str, Any] = Field(default_factory=dict, description="Additional context")
    
    # Performance settings
    timeout: int = Field(default=settings.GROQ_TIMEOUT, description="Request timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")

class LLMResponse(BaseModel):
    """
    Standardized response format from LLM operations.
    Contains generated content plus comprehensive metadata.
    """
    content: str = Field(..., description="Generated text content")
    model_used: str = Field(..., description="Model that generated the content")
    
    # Performance metrics
    generation_time: float = Field(..., description="Time taken for generation in seconds")
    tokens_used: int = Field(..., description="Number of tokens consumed")
    tokens_per_second: float = Field(..., description="Generation speed in tokens/second")
    
    # Quality indicators
    confidence_estimate: float = Field(default=0.8, ge=0.0, le=1.0, description="Estimated confidence")
    completion_reason: str = Field(default="stop", description="Why generation stopped")
    
    # Metadata
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: str = Field(..., description="Unique request identifier")
    
    # Cost tracking (estimated)
    estimated_cost: float = Field(default=0.0, description="Estimated API cost in USD")

class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.
    Defines the interface that all LLM providers must implement.
    """
    
    def __init__(self, provider_name: str):
        self.provider_name = provider_name
        self.logger = structlog.get_logger(f"llm_provider.{provider_name}")
        self.request_count = 0
        self.total_tokens_used = 0
        self.total_cost = 0.0
    
    @abstractmethod
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate content using the LLM provider"""
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the provider is healthy and accessible"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get provider usage statistics"""
        return {
            "provider": self.provider_name,
            "total_requests": self.request_count,
            "total_tokens": self.total_tokens_used,
            "total_cost": self.total_cost,
            "avg_cost_per_request": self.total_cost / max(self.request_count, 1)
        }

class GroqProvider(LLMProvider):
    """
    Groq LLM provider implementation using the Groq API.
    Handles all Groq-specific operations, error handling, and optimization.
    """
    
    def __init__(self, api_key: str):
        super().__init__("groq")
        self.client = AsyncGroq(api_key=api_key)
        
        # Model-specific pricing (approximate cents per 1K tokens)
        self.pricing = {
            "llama-3.1-70b-versatile": {"input": 0.059, "output": 0.079},
            "llama-3.1-8b-instant": {"input": 0.005, "output": 0.008},
            "mixtral-8x7b-32768": {"input": 0.024, "output": 0.024}
        }
        
        # Performance tracking
        self.model_performance = {}
        
        self.logger.info("Groq provider initialized", api_key_present=bool(api_key))
    
    async def generate(self, request: LLMRequest) -> LLMResponse:
        """
        Generate content using Groq's API with comprehensive error handling,
        retries, and performance tracking.
        """
        request_id = f"groq_{int(time.time() * 1000)}"
        start_time = time.time()
        
        self.logger.info(
            "Starting content generation",
            request_id=request_id,
            model=request.model,
            prompt_length=len(request.prompt),
            temperature=request.temperature
        )
        
        # Build messages for Groq API
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        messages.append({"role": "user", "content": request.prompt})
        
        # Attempt generation with retries
        last_exception = None
        for attempt in range(request.retry_attempts):
            try:
                # Make API call with timeout
                completion = await asyncio.wait_for(
                    self.client.chat.completions.create(
                        model=request.model,
                        messages=messages,
                        temperature=request.temperature,
                        max_tokens=request.max_tokens,
                        top_p=request.top_p,
                        stream=False
                    ),
                    timeout=request.timeout
                )
                
                # Extract response data
                generated_content = completion.choices[0].message.content
                completion_reason = completion.choices[0].finish_reason
                
                # Calculate metrics
                generation_time = time.time() - start_time
                tokens_used = completion.usage.total_tokens if completion.usage else 0
                tokens_per_second = tokens_used / max(generation_time, 0.001)
                
                # Estimate cost
                estimated_cost = self._estimate_cost(request.model, tokens_used)
                
                # Update statistics
                self.request_count += 1
                self.total_tokens_used += tokens_used
                self.total_cost += estimated_cost
                
                # Track model performance
                self._update_model_performance(request.model, generation_time, tokens_used)
                
                # Create response
                response = LLMResponse(
                    content=generated_content,
                    model_used=request.model,
                    generation_time=generation_time,
                    tokens_used=tokens_used,
                    tokens_per_second=tokens_per_second,
                    completion_reason=completion_reason,
                    request_id=request_id,
                    estimated_cost=estimated_cost
                )
                
                self.logger.info(
                    "Content generation successful",
                    request_id=request_id,
                    generation_time=generation_time,
                    tokens_used=tokens_used,
                    tokens_per_second=tokens_per_second,
                    attempt_number=attempt + 1
                )
                
                return response
                
            except asyncio.TimeoutError:
                last_exception = Exception(f"Request timed out after {request.timeout} seconds")
                self.logger.warning(
                    "Generation request timed out",
                    request_id=request_id,
                    attempt=attempt + 1,
                    timeout=request.timeout
                )
            
            except Exception as e:
                last_exception = e
                self.logger.warning(
                    "Generation attempt failed",
                    request_id=request_id,
                    attempt=attempt + 1,
                    error=str(e),
                    error_type=type(e).__name__
                )
            
            # Wait before retry (exponential backoff)
            if attempt < request.retry_attempts - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
        
        # All attempts failed
        self.logger.error(
            "Content generation failed after all retries",
            request_id=request_id,
            total_attempts=request.retry_attempts,
            final_error=str(last_exception)
        )
        
        raise Exception(f"Generation failed after {request.retry_attempts} attempts: {last_exception}")
    
    async def health_check(self) -> bool:
        """
        Verify that the Groq API is accessible and responsive.
        Uses a minimal test request to check connectivity.
        """
        try:
            test_request = LLMRequest(
                prompt="Hello",
                model=settings.GROQ_MODEL_FAST,  # Use fastest model for health check
                max_tokens=5,
                timeout=10
            )
            
            response = await self.generate(test_request)
            
            self.logger.info(
                "Groq health check successful",
                response_time=response.generation_time,
                model=response.model_used
            )
            
            return True
            
        except Exception as e:
            self.logger.error(
                "Groq health check failed",
                error=str(e),
                error_type=type(e).__name__
            )
            return False
    
    def _estimate_cost(self, model: str, tokens_used: int) -> float:
        """Estimate the cost of the API call based on model and tokens"""
        if model not in self.pricing:
            return 0.0
        
        # Rough estimation assuming 50/50 input/output split
        input_tokens = tokens_used * 0.5
        output_tokens = tokens_used * 0.5
        
        pricing = self.pricing[model]
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1000
        
        return cost
    
    def _update_model_performance(self, model: str, generation_time: float, tokens_used: int):
        """Track performance metrics by model"""
        if model not in self.model_performance:
            self.model_performance[model] = {
                "total_requests": 0,
                "total_time": 0.0,
                "total_tokens": 0,
                "avg_time": 0.0,
                "avg_tokens_per_second": 0.0
            }
        
        perf = self.model_performance[model]
        perf["total_requests"] += 1
        perf["total_time"] += generation_time
        perf["total_tokens"] += tokens_used
        perf["avg_time"] = perf["total_time"] / perf["total_requests"]
        perf["avg_tokens_per_second"] = perf["total_tokens"] / perf["total_time"]
    
    def get_model_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance statistics by model"""
        return self.model_performance.copy()

class LLMService:
    """
    High-level LLM service that manages multiple providers and provides
    intelligent routing, caching, and optimization features.
    """
    
    def __init__(self):
        self.providers: Dict[str, LLMProvider] = {}
        self.default_provider = "groq"
        self.logger = structlog.get_logger("llm_service")
        
        # Performance optimization
        self.response_cache: Dict[str, LLMResponse] = {}
        self.cache_hits = 0
        self.cache_misses = 0
        
        # Initialize providers
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available LLM providers based on configuration"""
        if settings.GROQ_API_KEY:
            try:
                groq_provider = GroqProvider(settings.GROQ_API_KEY)
                self.providers["groq"] = groq_provider
                self.logger.info("Groq provider initialized successfully")
            except Exception as e:
                self.logger.error("Failed to initialize Groq provider", error=str(e))
        
        if not self.providers:
            raise Exception("No LLM providers could be initialized. Check your API keys.")
    
    async def generate_content(
        self,
        prompt: str,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate content using the best available provider and model.
        Includes intelligent provider selection, caching, and error recovery.
        """
        # Determine provider and model
        provider_name = provider or self.default_provider
        model_name = model or self._select_optimal_model(prompt, kwargs.get('task_type'))
        
        # Create request
        request = LLMRequest(
            prompt=prompt,
            model=model_name,
            **kwargs
        )
        
        # Check cache first
        cache_key = self._generate_cache_key(request)
        if cache_key in self.response_cache:
            self.cache_hits += 1
            cached_response = self.response_cache[cache_key]
            self.logger.info(
                "Cache hit for content generation",
                cache_key=cache_key[:16] + "...",
                model=model_name
            )
            return cached_response
        
        self.cache_misses += 1
        
        # Get provider
        if provider_name not in self.providers:
            raise ValueError(f"Provider '{provider_name}' not available")
        
        llm_provider = self.providers[provider_name]
        
        # Generate content
        try:
            response = await llm_provider.generate(request)
            
            # Cache successful responses
            self.response_cache[cache_key] = response
            
            # Clean cache if it gets too large (simple LRU)
            if len(self.response_cache) > 100:
                oldest_key = next(iter(self.response_cache))
                del self.response_cache[oldest_key]
            
            return response
            
        except Exception as e:
            self.logger.error(
                "Content generation failed",
                provider=provider_name,
                model=model_name,
                error=str(e)
            )
            raise
    
    def _select_optimal_model(self, prompt: str, task_type: Optional[str] = None) -> str:
        """
        Select the most appropriate model based on prompt complexity and task type.
        This implements intelligent model routing for cost and performance optimization.
        """
        prompt_length = len(prompt)
        
        # Simple heuristics for model selection
        if task_type == "research" or prompt_length > 2000:
            return settings.GROQ_MODEL_SMART  # Use most capable model
        elif task_type == "qa" or prompt_length < 500:
            return settings.GROQ_MODEL_FAST   # Use fastest model
        elif task_type == "creative":
            return settings.GROQ_MODEL_CREATIVE  # Use creative model
        else:
            return settings.GROQ_MODEL_SMART  # Default to smart model
    
    def _generate_cache_key(self, request: LLMRequest) -> str:
        """Generate a cache key for the request"""
        import hashlib
        
        # Create hash from key request parameters
        key_data = f"{request.prompt}_{request.model}_{request.temperature}_{request.max_tokens}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all providers"""
        health_status = {}
        
        for provider_name, provider in self.providers.items():
            try:
                is_healthy = await provider.health_check()
                health_status[provider_name] = is_healthy
            except Exception as e:
                self.logger.error(f"Health check failed for {provider_name}", error=str(e))
                health_status[provider_name] = False
        
        return health_status
    
    def get_service_stats(self) -> Dict[str, Any]:
        """Get comprehensive service statistics"""
        stats = {
            "providers": {},
            "cache": {
                "hits": self.cache_hits,
                "misses": self.cache_misses,
                "hit_rate": self.cache_hits / max(self.cache_hits + self.cache_misses, 1),
                "cache_size": len(self.response_cache)
            },
            "total_providers": len(self.providers)
        }
        
        # Add provider-specific stats
        for provider_name, provider in self.providers.items():
            stats["providers"][provider_name] = provider.get_stats()
        
        return stats

# Global LLM service instance
llm_service = LLMService()
