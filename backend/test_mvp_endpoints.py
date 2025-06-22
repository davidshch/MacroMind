#!/usr/bin/env python3
"""
MVP Endpoint Test Script for Hackathon Demo
Tests the "Golden Path" endpoints from PROJECT_CONTEXT.md
"""

import asyncio
import aiohttp
import json
from datetime import datetime

# Test configuration
BASE_URL = "http://localhost:8888"
AUTH_TOKEN = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJ0ZXN0QGV4YW1wbGUuY29tIiwiZXhwIjoxNzUzMTQ1OTgxfQ.93qjnqS86kv2P_XU0Zct0opUiMJxbMSXuj_LxO4Bb3o"

HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

async def test_endpoint(session, method, endpoint, data=None, description=""):
    """Test a single endpoint and return results."""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method.upper() == "GET":
            async with session.get(url, headers=HEADERS) as response:
                result = await response.json()
        elif method.upper() == "POST":
            async with session.post(url, headers=HEADERS, json=data) as response:
                result = await response.json()
        
        success = response.status == 200
        return {
            "endpoint": endpoint,
            "method": method,
            "description": description,
            "status_code": response.status,
            "success": success,
            "response": result,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "endpoint": endpoint,
            "method": method,
            "description": description,
            "status_code": 0,
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

async def run_mvp_tests():
    """Run all MVP endpoint tests."""
    print("🚀 Starting MVP Endpoint Tests for Hackathon Demo")
    print("=" * 60)
    
    async with aiohttp.ClientSession() as session:
        tests = [
            # ACT I: The "What & Why" (Sentiment & LLM)
            ("GET", "/api/sentiment/AAPL", None, "Sentiment Analysis - Core MVP Endpoint"),
            ("GET", "/api/sentiment/TSLA", None, "Sentiment Analysis - Alternative Stock"),
            ("GET", "/api/sentiment/AAPL/insights", None, "LLM Insights - AI Explanation"),
            
            # ACT II: The "What's Next" (Proprietary ML)
            ("POST", "/api/volatility/predict", {
                "symbol": "AAPL",
                "lookback_days": 30,
                "prediction_days": 5
            }, "Volatility Prediction - Core ML Feature"),
            ("POST", "/api/volatility/predict", {
                "symbol": "TSLA",
                "lookback_days": 30,
                "prediction_days": 5
            }, "Volatility Prediction - Alternative Stock"),
            
            # ACT III: The "So What?" (Action)
            ("POST", "/api/alerts/", {
                "symbol": "AAPL",
                "conditions": {
                    "logical_operator": "AND",
                    "conditions": [
                        {
                            "metric": "volatility.predicted",
                            "operator": "GREATER_THAN",
                            "value": 0.3
                        }
                    ]
                },
                "notes": "High volatility alert for demo"
            }, "Alert Creation - Action System"),
        ]
        
        results = []
        for method, endpoint, data, description in tests:
            print(f"\n🔍 Testing: {description}")
            print(f"   {method} {endpoint}")
            
            result = await test_endpoint(session, method, endpoint, data, description)
            results.append(result)
            
            if result["success"]:
                print(f"   ✅ SUCCESS (Status: {result['status_code']})")
            else:
                print(f"   ❌ FAILED (Status: {result['status_code']})")
                if "error" in result:
                    print(f"   Error: {result['error']}")
        
        # Summary
        print("\n" + "=" * 60)
        print("📊 MVP ENDPOINT TEST SUMMARY")
        print("=" * 60)
        
        successful = sum(1 for r in results if r["success"])
        total = len(results)
        
        print(f"✅ Successful: {successful}/{total}")
        print(f"❌ Failed: {total - successful}/{total}")
        
        if successful == total:
            print("\n🎉 ALL MVP ENDPOINTS WORKING! READY FOR DEMO!")
        else:
            print("\n⚠️  SOME ENDPOINTS NEED ATTENTION")
            for result in results:
                if not result["success"]:
                    print(f"   ❌ {result['description']}: {result.get('error', 'Unknown error')}")
        
        # Save detailed results
        with open("mvp_test_results.json", "w") as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\n📄 Detailed results saved to: mvp_test_results.json")
        
        return results

if __name__ == "__main__":
    asyncio.run(run_mvp_tests()) 