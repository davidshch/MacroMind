# MacroMind Project

MacroMind is an AI-powered economic calendar designed to provide real-time market insights, sentiment analysis, and volatility predictions for traders, investors, and financial analysts. This project aims to deliver a comprehensive tool that integrates various data sources and utilizes advanced AI techniques to enhance decision-making in financial markets.

## Table of Contents

- [Core Features](#core-features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Core Features

- **Past Data Access**: Retrieve historical economic events and market data for analysis.
- **Real-Time Market Sentiment**: Analyze financial news and social media sentiment to detect market indicators.
- **AI-Powered Volatility Estimation**: Predict long-term volatility trends based on various economic factors.
- **Ticker Symbol Search**: Search for specific stock, forex, or crypto symbols with real-time data.
- **Event Tracking & Forecasting**: Track upcoming economic events and forecast their outcomes using AI.

## Tech Stack

- **Backend**: FastAPI (Python), PostgreSQL, Redis (optional), Hugging Face Transformers (BERT/GPT), Yahoo Finance API / Alpha Vantage, Stripe.
- **Frontend**: Next.js (React), Recharts.js, Axios.
- **Deployment**: Render / AWS EC2 (Backend), Vercel (Frontend).

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/macromind.git
   cd macromind
   ```

2. Set up the backend:
   - Navigate to the `backend` directory.
   - Install dependencies:
     ```
     pip install -r requirements.txt
     ```

3. Set up the frontend:
   - Navigate to the `frontend` directory.
   - Install dependencies:
     ```
     npm install
     ```

## Usage

- Start the backend server:
  ```
  uvicorn src.main:app --reload
  ```

- Start the frontend application:
  ```
  npm run dev
  ```

Visit `http://localhost:3000` to access the application.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or features you'd like to add.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.