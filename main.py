import ccxt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Conv1D, Dense, Flatten, Dropout
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
print(df)
# Use the closing price for the prediction
price_data = df['close'].values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(price_data)

# Create a time-series dataset
def create_dataset(data, window_size):
    X, y = [], []
    for i in range(len(data) - window_size - 1):
        X.append(data[i:(i + window_size), 0])
        y.append(data[i + window_size -1, 0])
    return np.array(X), np.array(y)

window_size = 60
X, y = create_dataset(scaled_data, window_size)

# Reshape the input data for 1D-CNN
X = np.reshape(X, (X.shape[0], X.shape[1], 1))

train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define the model
model = Sequential([
    Conv1D(filters=32, kernel_size=5, activation='relu', input_shape=(window_size, 1)),
    Flatten(),
    Dense(64, activation='relu'),
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
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
y_pred_actual = scaler.inverse_transform(y_pred)

# Calculate the mean squared error, R^2 score, and mean absolute error on the test set
test_loss = model.evaluate(X_test, y_test, verbose=0)
test_r2_score = r2_score(y_test_actual, y_pred_actual)
test_mae = mean_absolute_error(y_test_actual, y_pred_actual)

print(f"Test Loss (MSE): {test_loss}")
print(f"Test R^2 Score: {test_r2_score}")
print(f"Test Mean Absolute Error: {test_mae}")
