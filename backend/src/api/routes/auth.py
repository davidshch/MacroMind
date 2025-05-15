from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
import logging
from ...database.database import get_db
from ...services.user import UserService
from ...services.auth import verify_password, create_access_token
from ...schemas.auth import LoginRequest, RegisterRequest, TokenResponse

router = APIRouter()
logger = logging.getLogger(__name__)

@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """Register a new user and return an access token."""
    try:
        logger.debug(f"Attempting to register user with email: {request.email}")
        user_service = UserService(db)
        
        # Check if user exists
        existing_user = await user_service.get_user_by_email(request.email)
        if existing_user:
            logger.warning(f"Registration failed: Email already exists: {request.email}")
            raise HTTPException(status_code=400, detail="Email already registered")
        
        # Create user
        logger.debug("Creating new user...")
        user = await user_service.create_user(request.email, request.password)
        if not user:
            logger.error("User creation failed")
            raise HTTPException(status_code=500, detail="Failed to create user")
            
        # Create token
        logger.debug("Creating access token...")
        token = create_access_token({"sub": user.email})
        
        logger.info(f"User registered successfully: {request.email}")
        return TokenResponse(access_token=token, token_type="bearer")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration failed with error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: AsyncSession = Depends(get_db)):
    """Authenticate a user and return an access token."""
    try:
        logger.debug(f"Attempting to login user with email: {request.email}")
        user_service = UserService(db)
        
        # Get user by email
        user = await user_service.get_user_by_email(request.email)
        if not user:
            logger.warning(f"Login failed: User not found: {request.email}")
            raise HTTPException(status_code=404, detail="User not found")
        
        logger.debug(f"Found user, stored hash: {user.hashed_password[:10]}...")  # Log part of the hash
        
        # Verify password
        is_valid = verify_password(request.password, user.hashed_password)
        logger.debug(f"Password verification result: {is_valid}")
        
        if not is_valid:
            logger.warning(f"Login failed: Invalid password for user: {request.email}")
            raise HTTPException(status_code=401, detail="Invalid password")
        
        # Create token
        logger.debug("Creating access token...")
        token = create_access_token({"sub": user.email})
        
        logger.info(f"User logged in successfully: {request.email}")
        return TokenResponse(access_token=token, token_type="bearer")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login failed with error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")