# MacroMind Backend Documentation

## Overview
MacroMind is an AI-powered economic calendar that provides real-time market insights, sentiment analysis, and volatility predictions for traders, investors, and financial analysts. This backend application is built using FastAPI and serves as the API layer for the MacroMind project.

## Project Structure
```
macromind
├── backend
│   ├── src
│   │   ├── main.py                # Entry point of the backend application
│   │   ├── api                    # API package
│   │   │   ├── __init__.py        # Initializes the API package
│   │   │   ├── routes              # Contains route definitions
│   │   │   │   ├── auth.py        # User authentication routes
│   │   │   │   ├── events.py      # Economic events routes
│   │   │   │   ├── market_data.py # Real-time market data routes
│   │   │   │   └── sentiment.py    # Market sentiment analysis routes
│   │   ├── models                  # Database models
│   │   │   ├── __init__.py        # Initializes the models package
│   │   │   ├── event.py           # Event model
│   │   │   ├── sentiment.py       # Sentiment model
│   │   │   └── user.py            # User model
│   │   ├── services                # Business logic services
│   │   │   ├── __init__.py        # Initializes the services package
│   │   │   ├── market_data.py     # Market data processing functions
│   │   │   └── sentiment_analysis.py # Sentiment analysis functions
│   │   └── utils                   # Utility functions
│   │       └── __init__.py        # Initializes the utils package
│   ├── tests                       # Unit tests for the backend application
│   ├── requirements.txt            # Backend dependencies
│   └── README.md                   # Documentation for the backend application
├── frontend                        # Frontend application
└── README.md                       # Overall project documentation
```

## Getting Started
1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd macromind/backend
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**:
   ```bash
   uvicorn src.main:app --reload
   ```

## API Documentation
Refer to the individual route files in the `src/api/routes` directory for detailed information on available endpoints and their usage.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License
This project is licensed under the MIT License. See the LICENSE file for details.