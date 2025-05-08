# MacroMind API Documentation

## Volatility and Sentiment Analysis Endpoints

### GET /api/volatility/{symbol}
Retrieves detailed volatility predictions and market condition analysis for a given symbol.

**Parameters:**
- `symbol` (path): Stock/crypto symbol to analyze
- `lookback_days` (query, optional): Historical data window (default: 30)
- `prediction_days` (query, optional): Future prediction window (default: 5)

**Response:**
```json
{
    "symbol": "AAPL",
    "timestamp": "2025-04-27T10:30:00Z",
    "current_volatility": 0.15,
    "historical_volatility_annualized": 0.24,
    "volatility_10d_percentile": 75.5,
    "predicted_volatility": 0.18,
    "prediction_range": {
        "low": 0.16,
        "high": 0.20
    },
    "market_conditions": "volatile",
    "is_high_volatility": true,
    "trend": "increasing"
}
```

### GET /api/volatility/market-condition/{symbol}
Returns simplified market condition assessment.

**Parameters:**
- `symbol` (path): Stock/crypto symbol to analyze

**Response:**
```json
{
    "symbol": "AAPL",
    "market_condition": "highly_volatile",
    "is_high_volatility": true,
    "volatility_trend": "increasing",
    "current_percentile": 85.5,
    "timestamp": "2025-04-27T10:30:00Z"
}
```

### GET /api/sentiment/aggregated/{symbol}
Returns volatility-aware aggregated sentiment analysis with dynamic source weighting.

**Parameters:**
- `symbol` (path): Stock/crypto symbol to analyze
- `target_date` (query, optional): Target date (YYYY-MM-DD format, default: today)

**Response:**
```json
{
    "id": "unique_sentiment_record_id",
    "symbol": "AAPL",
    "date": "2025-04-27",
    "overall_sentiment": "bullish",
    "normalized_score": 0.68,
    "avg_daily_score": 0.68,
    "moving_avg_7d": 0.65,
    "benchmark": "QQQ",
    "news_sentiment": {
        "sentiment": "bullish",
        "confidence": 0.85,
        "sample_size": 20,
        "sentiment_distribution": {
            "bullish": 15,
            "bearish": 2,
            "neutral": 3
        },
        "last_updated": "2025-04-27T10:25:00Z"
    },
    "reddit_sentiment": {
        "sentiment": "bullish",
        "confidence": 0.65,
        "comment_count": 50,
        "last_updated": "2025-04-27T10:28:00Z"
    },
    "market_condition": "volatile",
    "volatility_context": {
        "level": 0.18,
        "is_high": true,
        "trend": "increasing"
    },
    "source_weights": {
        "news": 0.65,
        "reddit": 0.35
    },
    "timestamp": "2025-04-27T10:30:00Z"
}
```

### POST /api/sentiment/analyze-text
Analyzes the sentiment of a provided text snippet.

**Request Body:**
```json
{
    "text": "This stock is going to the moon!"
}
```

**Response:**
```json
{
    "sentiment": "bullish",
    "confidence": 0.95,
    "timestamp": "2025-04-27T11:00:00Z",
    "original_text": "This stock is going to the moon!"
}
```

### GET /api/sentiment/social/{symbol}
Get sentiment analysis primarily from social media sources (currently Reddit).

**Parameters:**
- `symbol` (path): Stock/crypto symbol to analyze

**Response:**
```json
{
    "sentiment": "bullish",
    "confidence": 0.65,
    "comment_count": 50,
    "top_keywords": ["buy", "hold", "rocket"],
    "sentiment_over_time": [...],
    "last_updated": "2025-04-27T10:28:00Z"
}
```

## Market Conditions

The API uses the following market condition classifications:

- `highly_volatile`: High current volatility with increasing trend
- `volatile`: High current volatility
- `increasing_volatility`: Normal current volatility but increasing trend
- `low_volatility`: Below average volatility
- `normal`: Average volatility levels

## Source Weight Adjustments

Sentiment source weights are dynamically adjusted based on market conditions:

| Market Condition | News Weight | Reddit Weight |
|-----------------|-------------|---------------|
| Highly Volatile | 0.70 | 0.30 |
| Volatile | 0.65 | 0.35 |
| Increasing Volatility | 0.63 | 0.37 |
| Low Volatility | 0.55 | 0.45 |
| Normal | 0.60 | 0.40 |

## Error Responses

```json
{
    "detail": "Error message describing what went wrong"
}
```

Common error status codes:
- `400`: Bad Request - Invalid parameters
- `404`: Not Found - Symbol not found
- `500`: Internal Server Error - Processing error

## Rate Limits

- Basic tier: 100 requests per hour
- Premium tier: 1000 requests per hour
- Enterprise tier: Custom limits

## Volatility Prediction Model

The volatility prediction model uses:
- XGBoost regression
- Technical indicators (RSI, ATR)
- Rolling volatility windows (5d, 10d, 30d)
- Market condition classification
