"""Enhanced error handling middleware"""
from fastapi import Request, HTTPException
from fastapi.responses import JSONResponse
from app.core.exceptions import MultiAgentError
import traceback
import datetime
class ErrorHandlerMiddleware:
    """Enhanced error handling for API requests"""
    
    async def __call__(self, request: Request, call_next):
        try:
            return await call_next(request)
            
        except MultiAgentError as e:
            return JSONResponse(
                status_code=400,
                content={
                    "error": e.error_code or "MULTI_AGENT_ERROR",
                    "message": e.message,
                    "details": e.details,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
            
        except HTTPException:
            raise
            
        except Exception as e:
            # Log the full traceback for debugging
            logger.error(f"Unexpected error: {traceback.format_exc()}")
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "INTERNAL_ERROR",
                    "message": "An unexpected error occurred",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
