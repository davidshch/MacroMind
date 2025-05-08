from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from ...database.database import get_db
from ...services.user import UserService
from ...services.auth import verify_password, create_access_token
from ...schemas.auth import LoginRequest, RegisterRequest, TokenResponse

router = APIRouter(prefix="/api/auth", tags=["auth"])

@router.post("/register", response_model=TokenResponse)
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    user_service = UserService(db)
    if user_service.get_user_by_email(request.email):
        raise HTTPException(status_code=400, detail="Email already registered")
    
    user = user_service.create_user(request.email, request.password)
    token = create_access_token({"sub": user.email})
    return TokenResponse(access_token=token)

@router.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    user_service = UserService(db)
    user = user_service.get_user_by_email(request.email)
    
    if not user or not verify_password(request.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    token = create_access_token({"sub": user.email})
    return TokenResponse(access_token=token)