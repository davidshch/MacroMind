# MacroMind API Documentation

## Overview
MacroMind provides real-time market insights through a RESTful API and WebSocket connections. 
This documentation covers all available endpoints, authentication, and data models.

## Authentication
All API endpoints require JWT authentication. Include the token in your request header:
```bash
Authorization: Bearer <your_jwt_token>
```

## Core Endpoints

### Market Data
- `GET /api/market/stock/{symbol}`
  - Real-time stock data
  - Price, volume, change percentage
  
- `GET /api/market/profile/{symbol}`
  - Company profile and fundamentals
  - Market cap, sector, industry

### Sentiment Analysis
- `GET /api/sentiment/analysis/{symbol}`
  - AI-powered sentiment analysis
  - News and social media sentiment

- `GET /api/sentiment/social/{symbol}`
  - Reddit sentiment analysis
  - Community engagement metrics

### Volatility Predictions
- `GET /api/volatility/{symbol}`
  - Historical and predicted volatility
  - Market condition assessment

### Economic Calendar
- `GET /api/economic-calendar/events`
  - Upcoming economic events
  - Impact predictions

## WebSocket Streaming
Connect to real-time data streams:
```javascript
ws://your-domain/ws/{symbol}
```

## Rate Limiting
- Basic tier: 60 requests/minute
- VIP tier: 300 requests/minute

## Error Codes
- 401: Authentication required
- 403: Insufficient permissions
- 429: Rate limit exceeded
- 500: Internal server error
