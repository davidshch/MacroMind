from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi import Request, Response
from starlette.responses import JSONResponse

limiter = Limiter(key_func=get_remote_address)

def setup_rate_limiter(app):
    """Set up rate limiting for the FastAPI application."""
    app.state.limiter = limiter
    
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded) -> Response:
        return JSONResponse(
            {"error": "Rate limit exceeded"},
            status_code=429
        )
