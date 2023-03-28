import ccxt
import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

# Initialize Binance API
exchange = ccxt.binance({'enableRateLimit': True})

# Get historical data
symbol = 'BTC/ZAR'
timeframe = '1h'
since = exchange.parse8601('2020-01-01T00:00:00Z')
ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since)

# Convert to DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Calculate technical indicators
df['RSI'] = talib.RSI(df['close'])
df['MACD'], _, _ = talib.MACD(df['close'])
df.dropna(inplace=True)

# Use the closing price, RSI, and MACD for the prediction
price_data = df[['close', 'RSI', 'MACD']].values

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(price_data)

# Create a time-series dataset
def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size - 1):
        X.append(data[i:(i + window_size), :])
        y.append(data[i + window_size, 0])
    return np.array(X), np.array(y)

window_size = 60
X, y = create_dataset(scaled_data, window_size)

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(window_size, X.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=True),
    Dropout(0.2),
    LSTM(50),
    Dropout(0.2),
    Dense(1)
])

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1, validation_split=0.2)

# Predict the target values for the test dataset
y_pred = model.predict(X_test)

# Invert scaling to obtain the actual price values
y_test_actual = scaler.inverse_transform(np.hstack((y_test.reshape(-1, 1), np.zeros((y_test.shape[0], 2))))[:, 0])
y_pred_actual = scaler.inverse_transform(np.hstack((y_pred, np.zeros((y_pred.shape[0], 2))))[:, 0])

test_loss = model.evaluate(X_test, y_test, verbose=0)
test_r2_score = r2_score(y_test_actual, y_pred_actual)
test_mae = mean_absolute_error(y_test_actual, y_pred_actual)

print(f"Test Loss (MSE): {test_loss}")
print(f"Test R^2 Score: {test_r2_score}")
print(f"Test Mean Absolute Error: {test_mae}")