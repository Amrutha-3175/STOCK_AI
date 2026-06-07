# AI Stock Predictor

## Overview

AI Stock Predictor is a machine learning-based stock analysis dashboard built using Python and Streamlit.

The application downloads real stock market data using Yahoo Finance, generates technical indicators, trains machine learning models, and provides stock price forecasts along with buy/sell signals.

## Features

* Real-time historical stock data using Yahoo Finance
* Stock selection (AAPL, MSFT, GOOGL, TSLA, NVDA)
* Technical Indicators:

  * Moving Average (MA_3)
  * Moving Average (MA_5)
  * Volatility
  * RSI (Relative Strength Index)
  * EMA (Exponential Moving Average)
* Machine Learning Models:

  * Linear Regression
  * Decision Tree Regressor
* Model comparison using MAE and RMSE
* Buy/Sell signal generation
* Backtesting simulation
* Interactive Streamlit dashboard

## Technologies Used

* Python
* Streamlit
* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Yahoo Finance API (yfinance)

## Project Workflow

1. Download historical stock data.
2. Generate technical indicators.
3. Split data into training and testing sets.
4. Train machine learning models.
5. Evaluate performance using MAE and RMSE.
6. Generate stock forecasts.
7. Display results in an interactive dashboard.

## Installation

```bash
pip install -r requirements.txt
```

Run the application:

```bash
python -m streamlit run app.py
```

## Future Improvements

* Additional technical indicators
* More stock choices
* Portfolio management module
* News sentiment analysis
* Advanced forecasting models

## Author

Amrutha V
