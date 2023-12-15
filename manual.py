import numpy as np
import requests
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import GRU, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from ta.trend import MACD
from ta.volatility import BollingerBands

# Function to fetch historical trading volume data
def fetch_volume_data(API_KEY, SYMBOL, INTERVAL, DATA_POINTS_TO_FETCH, DATA_POINTS_PER_REQUEST):
    volume_data = []
    start_time = 0

    while len(volume_data) < DATA_POINTS_TO_FETCH:
        data_points_to_fetch = min(DATA_POINTS_TO_FETCH - len(volume_data), DATA_POINTS_PER_REQUEST)
        url = f'https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval={INTERVAL}&limit={data_points_to_fetch}&startTime={start_time}'

        response = requests.get(url, headers={'X-MBX-APIKEY': API_KEY})

        if response.status_code == 200:
            new_data = response.json()
            if new_data:
                volume_data += new_data
                start_time = int(new_data[-1][0]) + 1
                print(f'Fetched {len(new_data)} new volume data points. Total: {len(volume_data)} data points.')
            else:
                print("No more volume data available.")
                break
        else:
            print("Failed to fetch volume data. Status code:", response.status_code)
            break

    return volume_data

# Binance API configuration
API_KEY = 'dXOWUeDdo4TJNuITBPfOLTfCgZvQiW8vn6W3M8iMxq14BUzvKBGpUdzSYDVeBy4x'
SYMBOL = 'BTCUSDT'
DATA_POINTS_TO_FETCH = 100000  # Total number of data points you want
DATA_POINTS_PER_REQUEST = 1000  # Number of data points per API request
INTERVAL = "5m"

# Fetch historical trading volume data
volume_data = fetch_volume_data(API_KEY, SYMBOL, INTERVAL, DATA_POINTS_TO_FETCH, DATA_POINTS_PER_REQUEST)

# Extract the closing prices and timestamps
closing_prices = [float(data[4]) for data in volume_data]
timestamps = [int(data[0]) for data in volume_data]

# Scale the data
scaler = MinMaxScaler()
closing_prices = scaler.fit_transform(np.array(closing_prices).reshape(-1, 1))

# Create a DataFrame
df = pd.DataFrame({'Timestamp': timestamps, 'Close': closing_prices.flatten()})

# Additional features
df['MACD'] = MACD(df['Close']).macd()
df['BB_High'] = BollingerBands(df['Close']).bollinger_hband()
df['BB_Low'] = BollingerBands(df['Close']).bollinger_lband()

# Prepare the data for training
X, y = [], []
for i in range(len(df) - 10):
    X.append(df.iloc[i:i+10][['Close', 'MACD', 'BB_High', 'BB_Low']])
    y.append(df.iloc[i+10]['Close'])

X = np.array(X)
y = np.array(y)

# Split the data into training and testing sets
split = int(0.8 * len(X))
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# Create a simple GRU model
model = Sequential()
model.add(GRU(32, activation='relu', input_shape=(10, X.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=25, batch_size=32)

# Make predictions on the test set
predictions = model.predict(X_test)

# Inverse transform the scaled predictions and actual values
predictions = scaler.inverse_transform(predictions)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Create an array of indices to match the length of predictions
indices = np.arange(len(y_test_actual))

# Visualize actual vs. predicted prices
plt.figure(figsize=(10, 6))
plt.plot(indices, y_test_actual, label='Actual Price (Test Data)', linestyle='-', marker='o')
plt.plot(indices, predictions, label='Predicted Price (Test Data)', linestyle='-', marker='o')
plt.xlabel('Data Point')
plt.ylabel('Price')
plt.legend()
plt.title(f'Actual vs. Predicted Price for {SYMBOL} (Test Data)')
plt.grid(True)
plt.show()
