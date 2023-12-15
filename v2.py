import csv
from binance.client import Client
import pandas as pd
import numpy as np
import time
import uuid
import matplotlib.pyplot as plt
from telegram import Bot
from decimal import Decimal, getcontext, ROUND_DOWN
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from hyperopt import fmin, tpe, hp
import threading
from statsmodels.tsa.statespace.sarimax import SARIMAX


# Binance API keys for the test environment
api_key = 'tTXlgT7hbCmQYf5c2tDkgKSXn7HfSXrd3uxOh9H5dxWcuzbUKLga2Adyd6oyUPeq'
api_secret = 'U8pbEJvyUwfmNuWGtnlrW2BN0r4TCuuOAGLQwFFt7VzNXtJNCt1WCKkXH9vYEi07'

# Create a Binance test client
client = Client(api_key, api_secret, testnet=True)

# Initialize the Telegram bot
telegram_bot_token = 'YOUR_BOT_TOKEN'
telegram_bot = Bot(token=telegram_bot_token)

# Lists to store transaction data
transactions = []
order_history = []  # Initialize the order history list

# Lists to store buy and sell points for plotting
buy_points = []  # Store buy points for plotting
sell_points = []  # Store sell points for plotting

# Define a function to predict future prices using ARIMA
from statsmodels.tsa.arima.model import ARIMA

# Function to predict future prices using ARIMA
def predict_arima(df, p, d, q):
    # Extract the historical closing prices
    prices = df['Close'].astype(float).tolist()

    # Fit the ARIMA model
    model = ARIMA(prices, order=(p, d, q))
    model_fit = model.fit()

    # Make predictions for the next 'n' time steps (adjust 'n' as needed)
    n = 5  # You can adjust this parameter
    forecast = model_fit.forecast(steps=n)

    return forecast.tolist()


space = {
    'sma_window': hp.choice('sma_window', [10, 20, 30, 40, 50]),
    'ema_window': hp.choice('ema_window', [5, 10, 15, 20, 25]),
    'macd_fast_period': hp.choice('macd_fast_period', [12, 14, 16, 18, 20]),
    'macd_slow_period': hp.choice('macd_slow_period', [26, 28, 30, 32, 34]),
    'macd_signal_period': hp.choice('macd_signal_period', [9, 10, 11, 12, 13]),
    'arima_p': hp.choice('arima_p', [1, 2, 3, 4, 5]),
    'arima_d': hp.choice('arima_d', [0, 1, 2, 3, 4]),
    'arima_q': hp.choice('arima_q', [0, 1, 2, 3, 4])
}


def predict_sarima(df, p, d, q, P, D, Q, s):
    try:
        # Extract the historical closing prices
        prices = df['Close'].astype(float).tolist()
        
        # Fit the SARIMA model
        model = SARIMAX(prices, order=(p, d, q), seasonal_order=(P, D, Q, s))
        model_fit = model.fit(disp=False)
        
        # Make predictions for the next 'n' time steps (adjust 'n' as needed)
        n = 5  # You can adjust this parameter
        forecast = model_fit.get_forecast(steps=n)
        forecast_mean = forecast.predicted_mean.tolist()
        
        return forecast_mean
        
    except Exception as e:
        print("Error predicting SARIMA:", e)
        return [None] * n  # Return a list of None if there's an error

# Define the objective function for optimization
def objective(params):
    sma_window = params['sma_window']
    ema_window = params['ema_window']
    macd_fast_period = params['macd_fast_period']
    macd_slow_period = params['macd_slow_period']
    macd_signal_period = params['macd_signal_period']
    arima_p = params['arima_p']
    arima_d = params['arima_d']
    arima_q = params['arima_q']

    # Run backtest with the given parameters and return a score (e.g., final profit)
    # You'll need to modify your backtesting logic to accept these parameters
    # and calculate the performance score based on your trading strategy.

    # For now, return a random score as a placeholder
    return np.random.rand()

# Perform hyperparameter optimization
best = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)

# Print the best parameter set
print("Best Parameters:", best)

# Extract the best hyperparameters
best_sma_window = [10, 20, 30, 40, 50][best['sma_window']]
best_ema_window = [5, 10, 15, 20, 25][best['ema_window']]
best_macd_fast_period = [12, 14, 16, 18, 20][best['macd_fast_period']]
best_macd_slow_period = [26, 28, 30, 32, 34][best['macd_slow_period']]
best_macd_signal_period = [9, 10, 11, 12, 13][best['macd_signal_period']]
best_arima_p = [1, 2, 3, 4, 5][best['arima_p']]
best_arima_d = [0, 1, 2, 3, 4][best['arima_d']]
best_arima_q = [0, 1, 2, 3, 4][best['arima_q']]

# Use the best hyperparameters for trading strategy
print("Best SMA Window:", best_sma_window)
print("Best EMA Window:", best_ema_window)
print("Best MACD Fast Period:", best_macd_fast_period)
print("Best MACD Slow Period:", best_macd_slow_period)
print("Best MACD Signal Period:", best_macd_signal_period)
print("Best ARIMA p:", best_arima_p)
print("Best ARIMA d:", best_arima_d)
print("Best ARIMA q:", best_arima_q)

def log_transaction(transaction, initial_balance, final_balance):
    with open('transaction_log.csv', 'a', newline='') as csvfile:
        fieldnames = ['Timestamp', 'Type', 'Quantity', 'Price', 'Initial Balance', 'Final Balance', 'SMA', 'EMA', 'RSI', 'MACD', 'Profit/Loss', 'Transaction Fee', 'Slippage', 'Cumulative Profit/Loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()  # Write header if the file is empty

        # Convert relevant values to Decimal objects
        transaction['Price'] = Decimal(str(transaction['Price']))
        transaction['Quantity'] = Decimal(str(transaction['Quantity']))

        # Calculate the initial and final balance for each transaction
        if transaction['Type'] == 'Buy':
            initial_balance -= transaction['Price'] * transaction['Quantity']
        elif transaction['Type'] == 'Sell':
            final_balance += transaction['Price'] * transaction['Quantity']

        # Calculate Profit/Loss
        transaction['Profit/Loss'] = final_balance - initial_balance

        # Transaction Fee (You need to replace this with actual fee calculation)
        transaction_fee = Decimal('0.0')  # Replace with actual fee calculation
        transaction['Transaction Fee'] = transaction_fee

        # Slippage (You need to replace this with actual slippage calculation)
        slippage = Decimal('0.0')  # Replace with actual slippage calculation
        transaction['Slippage'] = slippage

        # Cumulative Profit/Loss
        transaction['Cumulative Profit/Loss'] = final_balance - initial_balance

        writer.writerow(transaction)

def get_sma(df, window):
    sma = df['Close'].rolling(window=window).mean()
    return sma

def get_ema(df, window):
    ema = df['Close'].ewm(span=window, adjust=False).mean()
    return ema

# Function to calculate RSI
def calculate_rsi(data, window=14):
    delta = data.diff(1)
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD
def calculate_macd(data, fast_period=12, slow_period=26, signal_period=9):
    ema_fast = data.ewm(span=fast_period, adjust=False).mean()
    ema_slow = data.ewm(span=slow_period, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def send_telegram_message(message):
    try:
        # Replace 'YOUR_CHAT_ID' with the chat ID of the recipient
        chat_id = 'YOUR_CHAT_ID'

        # Send the message
        telegram_bot.send_message(chat_id=chat_id, text=message)
    except Exception as e:
        print("Error sending Telegram message:", e)

def set_precision(value, precision):
    getcontext().prec = precision
    return Decimal(value).quantize(Decimal('1E-' + str(precision)), rounding=ROUND_DOWN)

def execute_buy_order(symbol, quantity, price, initial_balance, final_balance, sma, ema, rsi, macd, sarima_forecast):
    try:
        # Generate a unique order ID using uuid
        order_id = str(uuid.uuid4())

        symbol_info = client.get_symbol_info(symbol)
        price_precision = int(symbol_info['quotePrecision'])
        quantity_precision = int(symbol_info['baseAssetPrecision'])

        quantity = set_precision(quantity, quantity_precision)
        price = set_precision(price, price_precision)

        order = client.create_test_order(
            symbol=symbol,
            side=Client.SIDE_BUY,
            type=Client.ORDER_TYPE_LIMIT,
            timeInForce=Client.TIME_IN_FORCE_GTC,
            quantity=quantity,
            price=price
        )

        transaction = {
            'Timestamp': pd.Timestamp.now(),
            'Type': 'Buy',
            'Quantity': quantity,
            'Price': price,
            'Initial Balance': initial_balance,
            'Final Balance': final_balance,
            'SMA': sma,
            'EMA': ema,
            'RSI': rsi,
            'MACD': macd,
            'SARIMA Forecast': sarima_forecast,
        }
        transactions.append(transaction)

        print("Buy order executed successfully.")
        print("Order ID:", order_id)
        order['customOrderId'] = order_id
        order_history.append(order)

    except Exception as e:
        print("Error executing buy order:", e)

def execute_sell_order(symbol, quantity, price, initial_balance, final_balance, sma, ema, rsi, macd, sarima_forecast):
    try:
        # Generate a unique order ID using uuid
        order_id = str(uuid.uuid4())

        # Get symbol information to determine precision
        symbol_info = client.get_symbol_info(symbol)
        price_precision = int(symbol_info['quotePrecision'])
        quantity_precision = int(symbol_info['baseAssetPrecision'])

        # Convert quantity and price to Decimal objects and set precision
        quantity = set_precision(quantity, quantity_precision)
        price = set_precision(price, price_precision)

        order = client.create_test_order(
            symbol=symbol,
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_LIMIT,
            timeInForce=Client.TIME_IN_FORCE_GTC,
            quantity=quantity,
            price=price
        )

        # Log the transaction
        transaction = {
            'Timestamp': pd.Timestamp.now(),
            'Type': 'Sell',
            'Quantity': quantity,
            'Price': price,
            'Initial Balance': initial_balance,
            'Final Balance': final_balance,
            'SMA': sma,
            'EMA': ema,
            'RSI': rsi,
            'MACD': macd,
            'SARIMA Forecast': sarima_forecast,
        }
        transactions.append(transaction)

        print("Sell order executed successfully.")
        print("Order ID:", order_id)
        order['customOrderId'] = order_id
        order_history.append(order)

    except Exception as e:
        print("Error executing sell order:", e)

def start_trading():
    try:
        # Define the trading symbol
        symbol = 'BTCUSDT'

        # Define the initial balance (You need to replace this with your actual initial balance)
        initial_balance = Decimal('10000.0')
        final_balance = initial_balance  # Initialize final_balance

        # Define parameters for SMA, EMA, RSI, MACD, and SARIMA
        sma_window = 20
        ema_window = 10
        macd_fast_period = 12
        macd_slow_period = 26
        macd_signal_period = 9
        arima_p = 1
        arima_d = 1
        arima_q = 0

        # Main trading loop
        while True:
            # Fetch historical data from Binance
            klines = client.futures_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1HOUR, limit=100)
            df = pd.DataFrame(klines, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
            df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
            df.set_index('Timestamp', inplace=True)

            # Calculate SMA and EMA
            df['SMA'] = get_sma(df, sma_window)
            df['EMA'] = get_ema(df, ema_window)

            # Calculate RSI
            df['Close'] = df['Close'].astype(float)
            df['RSI'] = calculate_rsi(df['Close'])

            # Calculate MACD
            macd, signal = calculate_macd(df['Close'], macd_fast_period, macd_slow_period, macd_signal_period)
            df['MACD'] = macd
            df['Signal'] = signal

            # Predict future prices using ARIMA
            arima_forecast = predict_arima(df, arima_p, arima_d, arima_q)

            # Predict future prices using SARIMA
            # SARIMA parameters (You can replace these with the optimized values)
            p = 1
            d = 1
            q = 0
            P = 0
            D = 0
            Q = 0
            s = 0
            sarima_forecast = predict_sarima(df, p, d, q, P, D, Q, s)

            # Extract the latest values
            current_sma = df['SMA'].iloc[-1]
            current_ema = df['EMA'].iloc[-1]
            current_rsi = df['RSI'].iloc[-1]
            current_macd = df['MACD'].iloc[-1]
            current_sarima_forecast = sarima_forecast[-1]

            # You can implement your trading strategy here based on the indicators and forecasts

            # Example: Buy if SMA crosses above EMA and RSI > 70
            if current_sma > current_ema and current_rsi > 70:
                # Place a buy order with a fixed quantity (You can implement your own quantity strategy)
                buy_quantity = Decimal('0.01')  # You can replace this with your quantity strategy
                buy_price = df['Close'].iloc[-1]  # Buy at the current closing price
                execute_buy_order(symbol, buy_quantity, buy_price, initial_balance, final_balance, current_sma, current_ema, current_rsi, current_macd, current_sarima_forecast)

            # Example: Sell if SMA crosses below EMA or RSI < 30
            elif current_sma < current_ema or current_rsi < 30:
                # Place a sell order with a fixed quantity (You can implement your own quantity strategy)
                sell_quantity = Decimal('0.01')  # You can replace this with your quantity strategy
                sell_price = df['Close'].iloc[-1]  # Sell at the current closing price
                execute_sell_order(symbol, sell_quantity, sell_price, initial_balance, final_balance, current_sma, current_ema, current_rsi, current_macd, current_sarima_forecast)

            # Sleep for a while before the next iteration
            print("[+] Sleeping...")
            time.sleep(10)  # Sleep for 1 hour

    except Exception as e:
        print("Error in trading loop:", e)

def load_historical_data(symbol, interval):
    klines = client.futures_klines(symbol=symbol, interval=interval, limit=1000)  # Adjust limit as needed
    df = pd.DataFrame(klines, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Close Time', 'Quote Asset Volume', 'Number of Trades', 'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df.set_index('Timestamp', inplace=True)
    return df

# Function to continuously train the model
def train_model_thread():
    while True:
        try:
            # Load historical data for training (adjust the symbol and interval as needed)
            training_symbol = 'BTCUSDT'
            training_interval = Client.KLINE_INTERVAL_1DAY  # Daily data for training
            training_data = load_historical_data(training_symbol, training_interval)

            # Train your model here with the training_data
            # Example: model.train(training_data)

            # Sleep for a while before the next training iteration (adjust the interval as needed)
            time.sleep(20)  # Sleep for 24 hours
        except Exception as e:
            print("Error in model training loop:", e)

# Function to visualize model predictions
def visualize_predictions_thread():
    while True:
        try:
            # Load recent data for visualization (adjust the symbol and interval as needed)
            visualization_symbol = 'BTCUSDT'
            visualization_interval = Client.KLINE_INTERVAL_1HOUR  # Hourly data for visualization
            visualization_data = load_historical_data(visualization_symbol, visualization_interval)

            # Make model predictions on the visualization_data
            # Example: predictions = predict_arima(visualization_data, p, d, q)
            
            # Calculate predictions using your chosen model (e.g., ARIMA)
            p = 1
            d = 1
            q = 0
            predictions = predict_arima(visualization_data, p, d, q)

            # Plot the predictions along with actual prices
            plot_predictions(visualization_data, predictions)
            print("[+] Sleeping...")
            time.sleep(10)  # Sleep for 1 hour
        except Exception as e:
            print("Error in visualization loop:", e)


# Function to plot model predictions
def plot_predictions(df, predictions):
    # Plot the actual prices
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Close'], label='Actual Price', alpha=0.7)

    # Plot the model predictions
    plt.plot(df.index[-len(predictions):], predictions, label='Predicted Price', alpha=0.7)

    plt.title('Actual vs. Predicted Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    try:
        # Create threads for trading, model training, and visualization
        trading_thread = threading.Thread(target=start_trading)
        model_training_thread = threading.Thread(target=train_model_thread)
        visualization_thread = threading.Thread(target=visualize_predictions_thread)

        # Start the threads
        trading_thread.start()
        model_training_thread.start()
        visualization_thread.start()

        # Wait for all threads to finish (you can add more logic here as needed)
        trading_thread.join()
        model_training_thread.join()
        visualization_thread.join()

    except Exception as e:
        print("Error in main function:", e)

if __name__ == "__main__":
    # Start the trading bot with multi-threading
    main()
