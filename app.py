from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
import yfinance as yf

st.title("AI Stock Predictor")

# Load data
#df = pd.read_csv("stock_data.csv")
ticker = st.sidebar.selectbox(
    "Select Stock",
    ["AAPL", "MSFT", "GOOGL", "TSLA", "NVDA"]
)

df = yf.download(
    ticker,
    start="2020-01-01",
    end="2025-01-01"
)

df = df.reset_index()
# Flatten Yahoo Finance MultiIndex columns
if hasattr(df.columns, "levels"):
    df.columns = [col[0] if isinstance(col, tuple) else col for col in df.columns]

# Feature engineering
df['MA_3'] = df['Close'].rolling(window=3).mean()
df['Return'] = df['Close'].pct_change()
df['MA_5'] = df['Close'].rolling(window=5).mean()
df['Volatility'] = df['Close'].rolling(window=3).std()
delta = df['Close'].diff()

gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)

avg_gain = gain.rolling(window=14).mean()
avg_loss = loss.rolling(window=14).mean()

rs = avg_gain / avg_loss

df['RSI'] = 100 - (100 / (1 + rs))
df['EMA_10'] = df['Close'].ewm(
    span=10,
    adjust=False
).mean()
df = df.dropna()
st.metric("Rows Downloaded:", len(df))
# Model
# Features and target
X = df[['MA_3', 'MA_5', 'Return', 'Volatility','RSI','EMA_10']]
y = df['Close']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    shuffle=False
)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

lr_predictions = lr_model.predict(X_test)

lr_mae = mean_absolute_error(
    y_test,
    lr_predictions
)

lr_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        lr_predictions
    )
)

# Decision Tree
tree_model = DecisionTreeRegressor(
    max_depth=5,
    random_state=42
)

tree_model.fit(X_train, y_train)

tree_predictions = tree_model.predict(X_test)

tree_mae = mean_absolute_error(
    y_test,
    tree_predictions
)

tree_rmse = np.sqrt(
    mean_squared_error(
        y_test,
        tree_predictions
    )
)

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["Linear Regression", "Decision Tree"]
)

if model_choice == "Linear Regression":
    predictions = lr_predictions
    mae = lr_mae
    rmse = lr_rmse
    model = lr_model
else:
    predictions = tree_predictions
    mae = tree_mae
    rmse = tree_rmse
    model = tree_model
latest_data = X.iloc[[-1]]

future_price = model.predict(latest_data)[0]

latest_actual = y.iloc[-1]

if future_price > latest_actual:
    forecast_signal = "BUY"

elif future_price < latest_actual:
    forecast_signal = "SELL"

else:
    forecast_signal = "HOLD"
# Evaluation Metrics
mae = mean_absolute_error(y_test, predictions)

mse = mean_squared_error(y_test, predictions)

rmse = np.sqrt(mse)



# Signal function
# Create test dataframe
test_df = X_test.copy()

test_df['Actual'] = y_test.values
test_df['Predicted'] = predictions

# Signal function
def get_signal(actual, predicted):
    diff = predicted - actual

    if diff > 1:
        return "STRONG BUY"
    elif diff > 0:
        return "BUY"
    elif diff < -1:
        return "STRONG SELL"
    else:
        return "HOLD"
    
st.subheader("Forecast")

st.success(
    f"Predicted Next Price: ${future_price:.2f}"
)

if forecast_signal == "BUY":
    st.success(f"Forecast Signal: {forecast_signal}")

elif forecast_signal == "SELL":
    st.error(f"Forecast Signal: {forecast_signal}")

else:
    st.warning(f"Forecast Signal: {forecast_signal}")

test_df['Signal'] = test_df.apply(
    lambda row: get_signal(row['Actual'], row['Predicted']),
    axis=1
)
# Simple Profit Simulation

test_df['Profit'] = 0.0
for i in range(len(test_df)-1):

    if test_df.iloc[i]['Signal'] in ['BUY', 'STRONG BUY']:

        current_price = test_df.iloc[i]['Actual']

        next_price = test_df.iloc[i+1]['Actual']

        test_df.loc[test_df.index[i], 'Profit'] = (
            next_price - current_price
        )
    total_profit = test_df['Profit'].sum()
# Show data
with st.expander("View Prediction Data"):
    st.dataframe(test_df)
st.subheader("Model Performance")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", f"{mae:.2f}")

with col2:
    st.metric("RMSE", f"{rmse:.2f}")

with col3:
    st.metric("Profit", f"{total_profit:.2f}")
# Show latest signal
st.subheader("Latest Signal")
st.subheader("Model Comparison")



st.write("Linear Regression MAE:", round(lr_mae, 2))
st.write("Decision Tree MAE:", round(tree_mae, 2))

st.write("Linear Regression RMSE:", round(lr_rmse, 2))
st.write("Decision Tree RMSE:", round(tree_rmse, 2))
st.subheader("Tomorrow Price Forecast")
st.subheader("Backtesting Results")

st.write(
    "Total Simulated Profit:",
    round(total_profit, 2)
)

st.success(
    f"Predicted Next Price: ₹{future_price:.2f}"
)

st.write(test_df[['Actual', 'Predicted', 'Signal']].tail(1))
# Plot

fig, ax = plt.subplots(figsize=(10,5))

ax.plot(y_test.values, label='Actual Price')

ax.plot(predictions, label='Predicted Price')

ax.set_title("Actual vs Predicted Stock Price")

ax.set_xlabel("Test Data Points")

ax.set_ylabel("Price")

ax.legend()

st.pyplot(fig)