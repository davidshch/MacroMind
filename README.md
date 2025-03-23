# MacroMind

Cutting-edge financial analytics platform and economic calendar combining AI-powered sentiment analysis with real-time market data.

## Overview

MacroMind helps traders and analysts make data-driven decisions by:
- Analyzing market sentiment across multiple sources in real-time
- Predicting volatility using advanced ML models
- Tracking economic events and their market impact
- Providing institutional-grade technical analysis

## Core Technology

- **AI-Powered Analysis**: FinBERT for sentiment analysis, XGBoost and Prophet for volatility forecasting
- **Real-Time Processing**: WebSocket streaming, efficient data aggregation
- **Multi-Source Data**: Integration with major financial APIs and social platforms
- **Advanced Analytics**: Technical indicators, sentiment correlation, market impact prediction

## Key Features

🔹 **Market Intelligence**
- Real-time sentiment analysis
- Economic event tracking
- Technical indicator analysis

🔹 **Advanced Analytics**
- Volatility predictions
- Market trend analysis
- Impact forecasting

🔹 **Technical Stack**
- Backend: FastAPI, PostgreSQL
- ML: Hugging Face, XGBoost, Facebook Prophet
- Data Sources: Alpha Vantage, Finnhub, News API, Reddit

## Project Status

✅ Implemented:
- **Core Infrastructure**
  - FastAPI backend setup with async support
  - PostgreSQL database integration
  - JWT authentication system
  - Rate limiting and caching mechanisms
  - Comprehensive error handling

- **Market Data Features**
  - Real-time stock data fetching (Alpha Vantage)
  - Enhanced company data (Finnhub)
  - Historical price data analysis
  - Technical indicator calculations
  - WebSocket streaming for live updates

- **AI & Analytics**
  - FinBERT-powered sentiment analysis
  - Multi-source sentiment aggregation (news, social)
  - Volatility analysis and forecasting
  - Market trend detection
  - Support/resistance level identification

- **Economic Calendar**
  - Event tracking and management
  - Impact prediction system
  - Market sector correlation analysis
  - Historical event performance tracking

- **Alert System**
  - Custom price alerts
  - Volatility threshold monitoring
  - Sentiment shift detection
  - Real-time notification system

- **VIP Features**
  - AI market explanation engine
  - Enhanced data access
  - Premium alert capabilities
  - Comprehensive market analysis

🚧 In Progress:
- Advanced sentiment analysis optimization
- Machine learning model refinements
- Testing coverage expansion
- Performance optimization

📋 Planned:
- Frontend development (Next.js)
- Advanced AI features integration
- Extended VIP capabilities
- Mobile responsiveness
- Additional data sources

## Technical Documentation

- [API Reference](./API_DOCUMENTATION.md)

## Security and Performance

- JWT authentication
- Rate limiting
- Async processing
- Efficient caching

## Architecture

```mermaid
graph TB
    subgraph External
        A[Market Data APIs] 
        B[News APIs]
        C[Social Media APIs]
    end
    
    subgraph Core Services
        D[Data Processing]
        E[Sentiment Analysis]
        F[Volatility Prediction]
    end
    
    subgraph Infrastructure
        G[FastAPI]
        H[PostgreSQL]
        I[WebSocket]
    end
    
    A & B & C --> D
    D --> E & F
    E & F --> G
    G --> H
    G --> I

```

## Setup Instructions
> Note: This project is currently in development. Core features are being implemented.

The full setup guide will be available once the MVP is completed.

## 📜 License

This project is licensed under the **Apache License 2.0**.  

You are free to use, modify, and distribute this software **as long as you comply with the license terms,** including **giving proper credit** and **not holding the developers liable for any use of this software.**  

📄 See the full **[LICENSE](LICENSE)** file for details.
