import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, LSTM
from binance.client import Client
from sklearn.preprocessing import MinMaxScaler

# Binance API keys for the test environment
api_key = 'AzXIe9u3YOVLdaHV9fFGRuV8SfTa2OWaMe0jQK7pDSTVYLqJJwsXokD1p03eoIK1'
api_secret = '7oYYHNrwOgIlZXdhe3ey4FZAQWWFIZvqufT1TtljtdVseqfmWhpOS8ulxL103Cl9'

# Initialize the Binance client
client = Client(api_key, api_secret)

# Global variables for the LSTM model and the last index
lstm_model = None
last_index = 0

# Global variables for trading parameters
buy_threshold = 0.02  # Buy when the forecasted price is 2% higher than the current price
sell_threshold = -0.02  # Sell when the forecasted price is 2% lower than the current price
quantity_btc = 0.0005  # Quantity in BTC to buy/sell
quantity = 0.1  # Quantity in USDT to buy/sell

def execute_buy_order(symbol, quantity, price):
    try:
        order = client.create_test_order(
            symbol=symbol,
            side=Client.SIDE_BUY,
            type=Client.ORDER_TYPE_LIMIT,
            timeInForce=Client.TIME_IN_FORCE_GTC,
            quantity=quantity,
            price=price
        )
        print("Buy order executed successfully.")
        print("Order Response:", order)
    except Exception as e:
        print("Error executing buy order:", e)
        

def execute_sell_order(symbol, quantity, price):
    try:
        order = client.create_test_order(
            symbol=symbol,
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_LIMIT,
            timeInForce=Client.TIME_IN_FORCE_GTC,
            quantity=quantity,
            price=price
        )
        print("Sell order executed successfully.")
        print("Order Response:", order)
    except Exception as e:
        print("Error executing sell order:", e)

def fit_lstm_model(data, look_back):
    try:
        X, y = [], []
        for i in range(len(data) - look_back):
            X.append(data[i:(i + look_back), 0])
            y.append(data[i + look_back, 0])
        X, y = np.array(X), np.array(y)

        model = Sequential()
        model.add(LSTM(50, input_shape=(look_back, 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(X, y, epochs=1, batch_size=1, verbose=2)
        return model
    except Exception as e:
        print("Error fitting LSTM model:", e)
        return None

def forecast_lstm_model(model, test_data):
    try:
        forecast = model.predict(test_data)
        return forecast[0][0]  # Return the forecasted value
    except Exception as e:
        print("Error forecasting with LSTM model:", e)
        return None

def plot_actual_vs_forecast(actual, forecast):
    try:
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(actual)), actual, label='Actual Data', color='blue')
        plt.plot(range(len(actual), len(actual) + len(forecast)), forecast, label='Forecast', color='red')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        #plt.show()
    except Exception as e:
        print("Error plotting actual vs. forecast:", e)

def save_to_csv(data, filename):
    try:
        df = pd.DataFrame(data, columns=['Price'])
        df.to_csv(filename, index=False)
        print(f"Data saved to {filename}.")
    except Exception as e:
        print("Error saving data to CSV:", e)

def forecast_price(symbol, interval, limit, look_back, future_periods):
    global lstm_model, last_index  # Declare lstm_model and last_index as global

    try:
        # Fetch more historical price data from Binance
        klines = client.futures_klines(symbol=symbol, interval=interval, limit=2 * limit)
        closing_prices = [float(candle[4]) for candle in klines]

        # Scale the data
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(np.array(closing_prices).reshape(-1, 1))

        # Fit the LSTM model only if it hasn't been fitted yet
        if lstm_model is None:
            lstm_model = fit_lstm_model(scaled_data, look_back)

        if lstm_model is not None:
            forecasts = []

            for i in range(future_periods):
                # Prepare data for forecasting
                lstm_input = scaled_data[last_index:last_index+look_back]
                lstm_input = np.reshape(lstm_input, (1, look_back, 1))  # Reshape the input for prediction

                # Make forecasts using the LSTM model
                lstm_forecast = forecast_lstm_model(lstm_model, lstm_input)

                if lstm_forecast is not None:
                    forecasts.append(lstm_forecast)
                    last_index += 1  # Update the last index
                else:
                    forecasts.append(np.nan)

            # Inverse transform to get original scale
            forecasts = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1))

            # Plot actual data vs. forecasted data
            plot_actual_vs_forecast(closing_prices, forecasts)

            # Save forecasted prices to a CSV file
            save_to_csv(forecasts, 'forecasted_prices.csv')

            # Return the last forecasted value
            return forecasts[-1][0]
        else:
            return None
    except Exception as e:
        print("Error forecasting price:", e)
        return None

def main():
    symbol = 'ETHUSDT'
    interval = Client.KLINE_INTERVAL_1HOUR
    limit = 100
    look_back = 10
    future_periods = 1

    while True:
        try:
            forecasted_price = forecast_price(symbol, interval, limit, look_back, future_periods)

            if forecasted_price is not None:
                print("Forecasted Price:", forecasted_price)

                # Implement your trading logic here based on the forecasted price
                if forecasted_price > buy_threshold:
                    execute_buy_order(symbol, quantity, forecasted_price)
                elif forecasted_price < sell_threshold:
                    execute_sell_order(symbol, quantity, forecasted_price)

            time.sleep(30)  # Sleep for 30 seconds before the next forecast
        except Exception as e:
            print("Error in main loop:", e)
            time.sleep(30)  # Sleep for 30 seconds before retrying

if __name__ == "__main__":
    main()
