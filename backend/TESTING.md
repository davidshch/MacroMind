# MacroMind Testing Guide

## Setup Testing Environment
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

# Install dependencies
pip install -r [requirements.txt](http://_vscodecontentref_/1)

# Set up test database
docker-compose up db -d

# Set up environment variables for testing
cp [.env.example](http://_vscodecontentref_/2) .env.test
# Edit .env.test with test credentials
```

## Running Tests

### Basic Test Commands
```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src tests/

# Run specific test file
pytest tests/test_api/test_market_data.py

# Run tests with detailed output
pytest -v

# Run tests by marker
pytest -m "integration"
```
### Coverage Reports
```bash
# Generate HTML coverage report
pytest --cov=src --cov-report=html tests/

# View report in browser
open htmlcov/index.html  # or manually open in browser
```

## Test Structure
```
tests/
├── conftest.py           # Shared fixtures
├── test_api/            # API endpoint tests
│   ├── test_auth.py
│   ├── test_market_data.py
│   └── test_sentiment.py
├── test_services/       # Service layer tests
│   ├── test_market_data.py
│   └── test_sentiment.py
└── test_models/        # Database model tests
    └── test_user.py
``` 

### Writing New Tests

## Test Template
```python
import pytest
from fastapi.testclient import TestClient

def test_feature_success(client: TestClient):
    """Test successful feature operation."""
    # Arrange
    test_data = {"key": "value"}
    
    # Act
    response = client.post("/api/endpoint", json=test_data)
    
    # Assert
    assert response.status_code == 200
    assert "expected_key" in response.json()

def test_feature_failure(client: TestClient):
    """Test feature error handling."""
    # Arrange
    invalid_data = {"invalid": "data"}
    
    # Act
    response = client.post("/api/endpoint", json=invalid_data)
    
    # Assert
    assert response.status_code == 400
```

## Example Test
```python
import pytest
from fastapi.testclient import TestClient

def test_market_data(client: TestClient):
    """Test market data retrieval for a stock symbol."""
    # Get market data for Apple stock
    response = client.get("/api/market/stock/AAPL")
    
    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert "price" in data
    assert "symbol" in data
    assert data["symbol"] == "AAPL"
```

## Key Features

- Automated unit tests using pytest
- Integration tests for API endpoints
- Database testing with PostgreSQL
- Coverage reporting for code quality


