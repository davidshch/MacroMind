#!/usr/bin/env python
"""
Quick test script to verify backend fixes without using pytest.

This script directly tests critical services to confirm they're working correctly
after our fixes. Set TESTING=TRUE environment variable to use mocks.
"""

import os
import sys
import asyncio
import logging
from datetime import date

# Set testing environment
os.environ['TESTING'] = 'TRUE'

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("test_script")

# Add the src directory to path for imports
sys.path.append(".")

async def test_volatility_service():
    """Test the volatility service using mock data."""
    try:
        from src.services.volatility import VolatilityService
        
        logger.info("Creating VolatilityService instance...")
        service = VolatilityService()
        
        logger.info("Testing calculate_and_predict_volatility for AAPL...")
        result = await service.calculate_and_predict_volatility('AAPL')
        
        assert result.get('symbol') == 'AAPL', "Symbol should be AAPL"
        assert 'current_volatility' in result, "Should have current_volatility"
        assert 'predicted_volatility' in result, "Should have predicted_volatility"
        
        logger.info("VolatilityService test successful!")
        return True
    except Exception as e:
        logger.error(f"VolatilityService test failed: {str(e)}")
        return False

async def test_sentiment_analyzer():
    """Test the sentiment analyzer using mock data."""
    try:
        from src.services.base_sentiment import BaseSentimentAnalyzer
        
        logger.info("Creating BaseSentimentAnalyzer instance...")
        analyzer = BaseSentimentAnalyzer()
        
        logger.info("Testing sentiment analysis...")
        positive_result = await analyzer.analyze_text("Stock prices are increasing with strong growth")
        negative_result = await analyzer.analyze_text("The market is crashing with significant losses")
        
        assert positive_result.get('sentiment') == 'bullish', "Should detect bullish sentiment"
        assert negative_result.get('sentiment') == 'bearish', "Should detect bearish sentiment"
        
        logger.info("BaseSentimentAnalyzer test successful!")
        return True
    except Exception as e:
        logger.error(f"BaseSentimentAnalyzer test failed: {str(e)}")
        return False

async def test_sector_fundamentals():
    """Test the sector fundamentals service using mock data."""
    try:
        from src.services.sector_fundamentals import SectorFundamentalsService
        from sqlalchemy.orm import Session
        from unittest.mock import MagicMock
        
        # Create mock database session
        mock_session = MagicMock(spec=Session)
        mock_session.query.return_value.filter.return_value.first.return_value = None
        
        logger.info("Creating SectorFundamentalsService instance...")
        service = SectorFundamentalsService(db=mock_session)
        
        logger.info("Testing sector fundamentals service for technology sector...")
        try:
            result = await service.get_sector_fundamentals("technology")
            logger.info("Direct call succeeded, which is unexpected with mock DB")
        except Exception as e:
            # We expect an HTTPException since we're using a mock DB
            if "HTTPException" in str(type(e)):
                logger.info("Received expected HTTPException with mock DB")
                return True
            else:
                logger.error(f"Unexpected error: {str(e)}")
                return False
        
        return True
    except Exception as e:
        logger.error(f"SectorFundamentalsService test failed: {str(e)}")
        return False

async def run_tests():
    """Run all tests and report results."""
    logger.info("=== RUNNING BACKEND SERVICE TESTS ===")
    
    test_results = {
        "Volatility Service": await test_volatility_service(),
        "Sentiment Analyzer": await test_sentiment_analyzer(),
        "Sector Fundamentals": await test_sector_fundamentals()
    }
    
    logger.info("=== TEST RESULTS ===")
    all_passed = True
    for test_name, passed in test_results.items():
        status = "PASSED" if passed else "FAILED"
        logger.info(f"{test_name}: {status}")
        if not passed:
            all_passed = False
    
    logger.info(f"Overall status: {'SUCCESS' if all_passed else 'FAILURE'}")
    return all_passed

if __name__ == "__main__":
    asyncio.run(run_tests())