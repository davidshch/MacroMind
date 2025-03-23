from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .rate_limit import RateLimiter
from typing import List

def setup_security(app: FastAPI, allowed_origins: List[str]):
    """Configure CORS, rate limiting and security headers."""
    
    # Set up CORS with stricter settings
    app.add_middleware(
        CORSMiddleware,
        allow_origins=allowed_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE"],
        allow_headers=["*"],
        max_age=3600
    )
    
    rate_limiter = RateLimiter()
    
    @app.middleware("http")
    async def security_middleware(request, call_next):
        # Rate limiting check
        await rate_limiter.check_rate_limit(request)
        
        response = await call_next(request)
        
        # Security headers
        response.headers.update({
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin"
        })
        
        return response
