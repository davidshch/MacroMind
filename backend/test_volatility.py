#!/usr/bin/env python
"""
Simplified test for the volatility service.
"""
import os
import asyncio
import logging

# Set testing environment
os.environ['TESTING'] = 'TRUE'

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("volatility_test")

async def test_volatility():
    try:
        from src.services.volatility import VolatilityService
        
        service = VolatilityService()
        result = await service.calculate_and_predict_volatility('AAPL')
        
        # Basic validation
        assert result['symbol'] == 'AAPL'
        assert 'current_volatility' in result
        assert 'predicted_volatility' in result
        
        print("✅ Volatility service test PASSED!")
        print(f"  - Current volatility: {result['current_volatility']}")
        print(f"  - Predicted volatility: {result['predicted_volatility']}")
        return True
    except Exception as e:
        print(f"❌ Volatility service test FAILED: {str(e)}")
        return False

if __name__ == "__main__":
    asyncio.run(test_volatility())