from sqlalchemy.orm import Session
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from ..database.models import User
from .auth import get_password_hash
from datetime import datetime
from typing import Optional

class UserService:
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_user(self, email: str, password: str) -> User:
        hashed_password = get_password_hash(password)
        user = User(
            email=email,
            hashed_password=hashed_password,
            created_at=datetime.utcnow()
        )
        self.db.add(user)
        await self.db.commit()
        await self.db.refresh(user)
        return user

    async def get_user_by_email(self, email: str) -> Optional[User]:
        result = await self.db.execute(
            select(User).filter(User.email == email)
        )
        return result.scalar_one_or_none()

    # Commenting out VIP-related logic in user service for MVP submission
    # async def update_vip_status(self, email: str, is_vip: bool) -> User:
    #     """Update user's VIP status."""
    #     user = await self.get_user_by_email(email)
    #     if not user:
    #         raise ValueError(f"User with email {email} not found")
    #     
    #     user.is_vip = is_vip
    #     await self.db.commit()
    #     await self.db.refresh(user)
    #     return user