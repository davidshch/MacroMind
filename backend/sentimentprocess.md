📊 MacroMind MVP AI Feature Implementation Guide

This file contains the critical AI and data processing tasks for the MacroMind MVP, based on the planning documents (Sentiment Process.pdf and Volatility Estimation.pdf).

⸻

📊 Sentiment Analysis Expansion (MVP Scope)

🔹 Goal

Enhance the Sentiment Analysis Pipeline to:
	•	Aggregate sentiment scores across News and Reddit.
	•	Dynamically adjust benchmarks based on market conditions.
	•	Collect basic sector-level fundamentals.
	•	Support multi-timeframe sentiment analysis (design only).

⸻

🔹 Required Features

1. Dynamic Benchmark Selection
	•	Choose SPY, QQQ, or RSP as the benchmark dynamically.
	•	Selection based on overall market conditions (e.g., volatility indicators).
	•	If market is very volatile, use RSP; otherwise default to SPY.

2. Sentiment Aggregation Pipeline
	•	Calculate the following for each day:
	•	Average daily sentiment score across Reddit and News.
	•	7-day moving average sentiment score.
	•	Separate tracking of Reddit sentiment and News sentiment.
	•	Normalize sentiment scores on a 0-1 or -1 to 1 scale.

3. Sector Fundamentals Collection
	•	For major sectors (e.g., Tech, Healthcare, Financials):
	•	P/E Ratio
	•	P/B Ratio
	•	Earnings Growth Rate
	•	Fetch using APIs:
	•	AlphaVantage API (or Finnhub if needed).
	•	Link each sector’s fundamentals to the overall sentiment database.

4. Multi-Timeframe Sentiment Framework
	•	Build structure to support analyzing sentiment over different horizons:
	•	Short-term: 1-5 days
	•	Medium-term: 5-30 days
	•	Long-term: 30+ days
	•	No need to fully implement timeframe logic yet — only modularize the code for future expansion.

📊 Volatility Prediction Model (MVP Scope)

🔹 Goal

Implement a basic AI model for volatility estimation:
	•	Forecast upcoming volatility for major indices based on past price movements and sentiment.

🔹 Required Features

1. Simple Volatility Estimation
	•	Use a model like Prophet or XGBoost to:
	•	Predict future realized volatility based on past historical volatility.
	•	Integrate aggregated sentiment score as an additional feature.

2. Basic Market Indicators
	•	Fetch market indicators needed for volatility model:
	•	Past historical prices (daily close)
	•	Basic VIX data if available (optional for MVP)

3. Beta Prediction (Defer)
	•	Do NOT implement dynamic beta prediction yet.
	•	Save beta prediction for post-MVP Phase 2.

📊 Infrastructure & Backend Requirements
	•	Extend sentiment_pipeline.py to support new aggregation logic.
	•	Add sector_fundamentals.py to fetch and store sector P/E, P/B, earnings growth.
	•	Modify database schema if necessary:
	•	Add fields for sector fundamentals.
	•	Add fields for multi-source sentiment.
	•	Create a new service volatility_model.py for volatility forecasting.
	•	Extend FastAPI routes to serve new sentiment aggregates and volatility forecasts.

📈 Notes for GitHub Copilot
	•	Prioritize implementing MVP features clearly.
	•	Use clean modular structure (separate data fetchers, processors, API routes).
	•	Document all major classes and functions with Python docstrings.
	•	Keep volatility and sentiment models in independent modules.
	•	Database models should evolve with minimal migrations if possible.

