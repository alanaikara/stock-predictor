# Nifty Trading Bot: Stock Performance Prediction

## Overview

This Nifty Trading Bot is a sophisticated Python-based stock analysis tool designed to predict and analyze the performance of Nifty 50 stocks using machine learning techniques. The bot fetches historical stock data, applies advanced feature engineering, and uses a Random Forest Regression model to predict potential stock performance.

## Features

- Fetch historical stock data for Nifty 50 stocks
- Calculate advanced financial indicators:
  - Returns
  - Simple Moving Average (SMA)
  - Exponential Moving Average (EMA)
  - Relative Strength Index (RSI)
  - Volatility
- Machine learning-based stock price prediction
- Risk-adjusted return analysis
- Export of recommended stocks to CSV

## Prerequisites

- Python 3.8+
- pip (Python package manager)

## Required Libraries

- pandas
- numpy
- yfinance
- scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/alanaikara/stock-predictor.git
cd stock_predictor
```

2. Create a virtual environment:
```bash
python3 -m venv stockbot_env
source stockbot_env/bin/activate  # On macOS/Linux
# stockbot_env\Scripts\activate  # On Windows
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

Run the main script:
```bash
python predict_tommorows_price.py
```

The script will:
1. Fetch historical stock data
2. Train prediction models
3. Analyze stock performances
4. Generate a CSV of recommended stocks

## Output

The script generates:
- Console output with stock performance rankings
- `recommended_stocks.csv` with detailed stock analysis

## How It Works

### Data Retrieval
- Fetches historical stock data for Nifty 50 stocks
- Covers stocks like Reliance, TCS, HDFC Bank, and more

### Feature Engineering
- Calculates advanced financial indicators
- Prepares data for machine learning

### Prediction Model
- Uses Random Forest Regression
- Predicts next-day closing prices
- Evaluates model performance using MSE and R-squared

### Analysis Metrics
- Potential return percentage
- Stock volatility
- Risk-adjusted return

## Limitations

- Predictions based on historical data
- No guarantee of future performance
- Simplified market analysis model

## Customization

You can:
- Modify stock tickers
- Adjust date ranges
- Add more features
- Experiment with different ML models

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Specify your license here, e.g., MIT]

## Disclaimer

This tool is for educational and research purposes. Always conduct thorough research and consult financial advisors before making investment decisions.

## Contact

Alan
