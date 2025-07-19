from pydantic_settings import BaseSettings
from typing import Optional, List
import os
from pathlib import Path

class Settings(BaseSettings):
    """
    Application configuration using Pydantic Settings.
    Manages all environment variables and application settings.
    """
    
    # Application Metadata
    APP_NAME: str = "Multi-Agent AI Content Generator"
    APP_VERSION: str = "2.0.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    RELOAD: bool = True
    WORKERS: int = 1
    
    # Groq LLM Configuration
    GROQ_API_KEY: Optional[str] = None
    GROQ_MODEL_FAST: str = "llama-3.1-8b-instant"      # For quick tasks
    GROQ_MODEL_SMART: str = "llama-3.1-70b-versatile"  # For complex tasks
    GROQ_MODEL_CREATIVE: str = "mixtral-8x7b-32768"    # For creative content
    GROQ_TEMPERATURE: float = 0.7
    GROQ_MAX_TOKENS: int = 1000
    GROQ_TIMEOUT: int = 30
    
    # Multi-Agent System Configuration
    MAX_CONCURRENT_AGENTS: int = 5
    AGENT_TIMEOUT: int = 45
    WORKFLOW_TIMEOUT: int = 180
    MAX_RETRY_ATTEMPTS: int = 3
    ENABLE_AGENT_MEMORY: bool = True
    
    # Content Generation Settings
    DEFAULT_WORD_COUNT: int = 200
    MIN_WORD_COUNT: int = 50
    MAX_WORD_COUNT: int = 2000
    CONTENT_QUALITY_THRESHOLD: float = 0.7
    
    # Performance & Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    CACHE_TTL_SECONDS: int = 300
    MAX_REQUEST_SIZE: int = 1024 * 1024  # 1MB
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./content_generator.db"
    DATABASE_ECHO: bool = False
    
    # Redis Configuration
    REDIS_URL: str = "redis://localhost:6379/0"
    REDIS_ENABLED: bool = False
    
    # Security
    SECRET_KEY: str = "swbkDYCh8yANLR72pXFlqwU5FXxAp2rB"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    CORS_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8501"]
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s | %(levelname)8s | %(name)s | %(message)s"
    ENABLE_RICH_LOGGING: bool = True
    LOG_FILE: Optional[str] = None
    
    # Monitoring & Analytics
    ENABLE_METRICS: bool = True
    ENABLE_TRACING: bool = True
    METRICS_PORT: int = 8001
    
    # File Upload Configuration
    UPLOAD_DIR: str = "uploads"
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    ALLOWED_EXTENSIONS: List[str] = [".txt", ".pdf", ".docx", ".md"]
    
    # WebSocket Configuration
    WS_HEARTBEAT_INTERVAL: int = 30
    WS_MAX_CONNECTIONS: int = 100
    
    # Development Settings
    RELOAD_DIRS: List[str] = ["app"]
    RELOAD_EXTENSIONS: List[str] = [".py"]
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Create necessary directories
        Path(self.UPLOAD_DIR).mkdir(exist_ok=True)

def validate_config() -> bool:
    """
    Validate critical configuration settings at application startup.
    Ensures all required settings are properly configured.
    """
    errors = []
    
    # Validate required API keys
    if not settings.GROQ_API_KEY:
        errors.append("GROQ_API_KEY is required. Get one from https://console.groq.com/")
    
    # Validate performance settings
    if settings.MAX_CONCURRENT_AGENTS > 10:
        errors.append("MAX_CONCURRENT_AGENTS should not exceed 10 for optimal performance")
    
    if settings.WORKFLOW_TIMEOUT < 60:
        errors.append("WORKFLOW_TIMEOUT should be at least 60 seconds")
    
    # Validate content settings
    if settings.MAX_WORD_COUNT > 5000:
        errors.append("MAX_WORD_COUNT should not exceed 5000 for performance reasons")
    
    # Validate security settings
    if settings.SECRET_KEY == "your-super-secret-key-change-in-production" and not settings.DEBUG:
        errors.append("SECRET_KEY must be changed in production environment")
    
    # Report validation results
    if errors:
        for error in errors:
            print(f"❌ Configuration Error: {error}")
        raise ValueError(f"Configuration validation failed: {len(errors)} errors found")
    
    print(f"✅ Configuration validated successfully")
    print(f"🤖 Multi-Agent System: {settings.MAX_CONCURRENT_AGENTS} concurrent agents")
    print(f"🚀 LLM Provider: Groq ({settings.GROQ_MODEL_SMART})")
    print(f"🌐 API Server: {settings.HOST}:{settings.PORT}")
    return True

# Global settings instance
settings = Settings()

# Validate configuration on import
if __name__ == "__main__":
    validate_config()
