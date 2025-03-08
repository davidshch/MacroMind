from sqlalchemy.orm import Session
from ..database.models import User
from .auth import get_password_hash
from datetime import datetime
from typing import Optional

class UserService:
    def __init__(self, db: Session):
        self.db = db

    def create_user(self, email: str, password: str) -> User:
        hashed_password = get_password_hash(password)
        user = User(
            email=email,
            hashed_password=hashed_password,
            created_at=datetime.utcnow()
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user

    def get_user_by_email(self, email: str) -> User:
        return self.db.query(User).filter(User.email == email).first()

    def update_vip_status(self, email: str, is_vip: bool) -> User:
        """Update user's VIP status."""
        user = self.get_user_by_email(email)
        if not user:
            raise ValueError(f"User with email {email} not found")
        
        user.is_vip = is_vip
        self.db.commit()
        self.db.refresh(user)
        return user
