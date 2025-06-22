#!/usr/bin/env python3
"""
Script to create a test user for MVP endpoint testing.
"""

import asyncio
import sys
import os
from datetime import datetime

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.database.database import get_db
from src.database.models import User
from src.services.auth import get_password_hash

async def create_test_user():
    """Create a test user for MVP testing."""
    
    async for db in get_db():
        try:
            # Check if test user already exists
            from sqlalchemy import select
            stmt = select(User).where(User.email == "test@example.com")
            result = await db.execute(stmt)
            existing_user = result.scalar_one_or_none()
            
            if existing_user:
                print(f"Test user already exists with ID: {existing_user.id}")
                return existing_user.id
            
            # Create new test user
            test_user = User(
                email="test@example.com",
                hashed_password=get_password_hash("testpassword123"),
                is_vip=False,
                created_at=datetime.now()
            )
            
            db.add(test_user)
            await db.commit()
            await db.refresh(test_user)
            
            print(f"Created test user with ID: {test_user.id}")
            return test_user.id
            
        except Exception as e:
            print(f"Error creating test user: {e}")
            await db.rollback()
            raise
        finally:
            await db.close()
        break

if __name__ == "__main__":
    user_id = asyncio.run(create_test_user())
    print(f"Test user ID: {user_id}") 