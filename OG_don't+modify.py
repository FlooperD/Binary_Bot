import csv
from binance.client import Client
import pandas as pd
import numpy as np
import time
import uuid
import matplotlib.pyplot as plt
from telegram import Bot
from decimal import Decimal, ROUND_DOWN

# Binance API keys for the test environment
api_key = 'tTXlgT7hbCmQYf5c2tDkgKSXn7HfSXrd3uxOh9H5dxWcuzbUKLga2Adyd6oyUPeq'
api_secret = 'U8pbEJvyUwfmNuWGtnlrW2BN0r4TCuuOAGLQwFFt7VzNXtJNCt1WCKkXH9vYEi07'

# Create a Binance test client
client = Client(api_key, api_secret, testnet=True)

# Initialize the Telegram bot
telegram_bot_token = '5500076996:AAHC1JSDHgJVDhDqTH6LbIJt783gGZM2HSs'
telegram_bot = Bot(token=telegram_bot_token)

# Lists to store transaction data
transactions = []
order_history = []  # Initialize the order history list

# Function to log transactions to a CSV file
def log_transaction(transaction, initial_balance, final_balance):
    with open('transaction_log.csv', 'a', newline='') as csvfile:
        fieldnames = ['Timestamp', 'Type', 'Quantity', 'Price', 'Initial Balance', 'Final Balance', 'SMA', 'EMA', 'RSI', 'MACD', 'Profit/Loss', 'Transaction Fee', 'Slippage', 'Cumulative Profit/Loss']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if csvfile.tell() == 0:
            writer.writeheader()  # Write header if the file is empty
        
        # Calculate the initial and final balance for each transaction
        if transaction['Type'] == 'Buy':
            initial_balance -= transaction['Price'] * transaction['Quantity']
        elif transaction['Type'] == 'Sell':
            final_balance += transaction['Price'] * transaction['Quantity']
        
        # Calculate Profit/Loss
        transaction['Profit/Loss'] = final_balance - initial_balance
        
        # Transaction Fee (You need to replace this with actual fee calculation)
        transaction_fee = 0.0  # Replace with actual fee calculation
        transaction['Transaction Fee'] = transaction_fee
        
        # Slippage (You need to replace this with actual slippage calculation)
        slippage = 0.0  # Replace with actual slippage calculation
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
        chat_id = '5044502086'

        # Send the message
        telegram_bot.send_message(chat_id=chat_id, text=message)
    except Exception as e:
        print("Error sending Telegram message:", e)

def execute_buy_order(symbol, quantity, price, initial_balance, final_balance, sma, ema, rsi, macd):
    try:
        # Generate a unique order ID using uuid
        order_id = str(uuid.uuid4())

        # Get symbol information to determine lot size rules
        symbol_info = client.get_symbol_info(symbol)
        min_qty = Decimal(symbol_info['filters'][1]['minQty'])
        step_size = Decimal(symbol_info['filters'][1]['stepSize'])

        # Convert quantity and price to Decimal objects
        quantity = Decimal(str(quantity))
        price = Decimal(str(price))

        # Round the quantity down to the nearest valid quantity using the step size
        rounded_quantity = (quantity / step_size).quantize(0, rounding=ROUND_DOWN) * step_size

        order = client.create_test_order(
            symbol=symbol,
            side=Client.SIDE_BUY,
            type=Client.ORDER_TYPE_LIMIT,
            timeInForce=Client.TIME_IN_FORCE_GTC,
            quantity=rounded_quantity,
            price=price  # No need to round price, as it's not subject to lot size rules
        )
        print("Buy order executed successfully.")
        print("Order ID:", order_id)  # Print your custom order ID
        order['customOrderId'] = order_id  # Add custom order ID to the order details
        order_history.append(order)  # Add the order details to order_history
        
        # Record the buy transaction and log it
        transaction = {
            'Timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'Type': 'Buy',
            'Quantity': rounded_quantity,
            'Price': price,
            'SMA': sma,
            'EMA': ema,
            'RSI': rsi,
            'MACD': macd
        }
        transactions.append(transaction)
        log_transaction(transaction, initial_balance, final_balance)
        
        # Send a Telegram message when a buy order is executed
        message = f"Buy order executed:\nSymbol: {symbol}\nQuantity: {rounded_quantity}\nPrice: {price}\nSMA: {sma}\nEMA: {ema}\nRSI: {rsi}\nMACD: {macd}\nInitial Balance: {initial_balance}\nFinal Balance: {final_balance}"
        send_telegram_message(message)
    except Exception as e:
        print("Error executing buy order:", e)

def execute_sell_order(symbol, quantity, price, initial_balance, final_balance, sma, ema, rsi, macd):
    try:
        # Generate a unique order ID using uuid
        order_id = str(uuid.uuid4())

        # Get symbol information to determine lot size rules
        symbol_info = client.get_symbol_info(symbol)
        min_qty = Decimal(symbol_info['filters'][1]['minQty'])
        step_size = Decimal(symbol_info['filters'][1]['stepSize'])

        # Convert quantity and price to Decimal objects
        quantity = Decimal(str(quantity))
        price = Decimal(str(price))

        # Round the quantity down to the nearest valid quantity using the step size
        rounded_quantity = (quantity / step_size).quantize(0, rounding=ROUND_DOWN) * step_size

        order = client.create_test_order(
            symbol=symbol,
            side=Client.SIDE_SELL,
            type=Client.ORDER_TYPE_LIMIT,
            timeInForce=Client.TIME_IN_FORCE_GTC,
            quantity=rounded_quantity,
            price=price  # No need to round price, as it's not subject to lot size rules
        )
        print("Sell order executed successfully.")
        print("Order ID:", order_id)  # Print your custom order ID
        order['customOrderId'] = order_id  # Add custom order ID to the order details
        order_history.append(order)  # Add the order details to order_history
        
        # Record the sell transaction and log it
        transaction = {
            'Timestamp': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()),
            'Type': 'Sell',
            'Quantity': rounded_quantity,
            'Price': price,
            'SMA': sma,
            'EMA': ema,
            'RSI': rsi,
            'MACD': macd
        }
        transactions.append(transaction)
        log_transaction(transaction, initial_balance, final_balance)
        
        # Send a Telegram message when a sell order is executed
        message = f"Sell order executed:\nSymbol: {symbol}\nQuantity: {rounded_quantity}\nPrice: {price}\nSMA: {sma}\nEMA: {ema}\nRSI: {rsi}\nMACD: {macd}\nInitial Balance: {initial_balance}\nFinal Balance: {final_balance}"
        send_telegram_message(message)
    except Exception as e:
        print("Error executing sell order:", e)

def calculate_profit_loss(initial_balance):
    balance = initial_balance
    total_buy_quantity = 0
    total_buy_cost = 0

    for transaction in transactions:
        if transaction['Type'] == 'Buy':
            total_buy_quantity += transaction['Quantity']
            total_buy_cost += transaction['Price'] * transaction['Quantity']
        elif transaction['Type'] == 'Sell':
            sell_quantity = transaction['Quantity']
            sell_price = transaction['Price']
            if total_buy_quantity > 0:
                avg_buy_price = total_buy_cost / total_buy_quantity
                profit_loss = (sell_price - avg_buy_price) * sell_quantity
                balance += profit_loss
                total_buy_quantity -= sell_quantity
                total_buy_cost -= avg_buy_price * sell_quantity

    return balance

def backtest_sma_ema_strategy(symbol, sma_window, ema_window, buy_threshold, sell_threshold, interval, initial_balance):
    try:
        historical_klines = client.futures_klines(symbol=symbol, interval=interval)
        
        df = pd.DataFrame(historical_klines, columns=['Time', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'NumberOfTrades', 'TakerBuyBaseAssetVolume', 'TakerBuyQuoteAssetVolume', 'Ignore'])
        df['Time'] = pd.to_datetime(df['Time'], unit='ms')
        df['Close'] = df['Close'].astype(float)
        
        df['SMA'] = get_sma(df, sma_window)
        df['EMA'] = get_ema(df, ema_window)
        
        holdings = 0
        buy_price = 0
        buy_points = []  # Store buy points for plotting
        sell_points = []  # Store sell points for plotting
        final_balance = initial_balance  # Initialize the final balance with the initial balance

        for i, row in df.iterrows():
            if i < max(sma_window, ema_window):
                continue

            rsi_window = 14  # RSI window
            macd_fast_period = 12  # MACD fast period
            macd_slow_period = 26  # MACD slow period
            macd_signal_period = 9  # MACD signal period

            # Calculate RSI
            rsi_data = df.loc[i - rsi_window + 1 : i, 'Close'].astype(float)
            rsi = calculate_rsi(rsi_data, rsi_window)

            # Calculate MACD
            macd_data = df.loc[i - max(macd_fast_period, macd_slow_period) + 1 : i, 'Close'].astype(float)
            macd, signal = calculate_macd(macd_data, macd_fast_period, macd_slow_period, macd_signal_period)

            if (
                row['Close'] > row['SMA']
                and row['Close'] > row['EMA']
                and rsi.iloc[-1] > 50
                and macd.iloc[-1] > signal.iloc[-1]
                and holdings == 0
            ):
                # Buy signal (when both SMA, EMA, RSI, and MACD are favorable)
                buy_price = row['Close']

                # Implement risk management (dynamic quantity and stop loss)
                balance_for_trade = min(initial_balance, final_balance)  # Use the lower of initial and final balance for risk management
                max_loss_percentage = 0.02  # Maximum allowed loss percentage (adjust as needed)
                stop_loss_price = buy_price * (1 - max_loss_percentage)
                max_quantity = balance_for_trade / stop_loss_price
                execute_buy_order(
                    symbol,
                    max_quantity,
                    buy_price,
                    initial_balance,
                    final_balance,
                    row['SMA'],
                    row['EMA'],
                    rsi.iloc[-1],
                    macd.iloc[-1]
                )
                holdings += max_quantity
                buy_points.append((row['Time'], buy_price))
            elif (
                (row['Close'] < row['SMA']
                or row['Close'] < row['EMA']
                or rsi.iloc[-1] < 30
                or macd.iloc[-1] < signal.iloc[-1])
                and holdings > 0
            ):
                # Sell signal (when either SMA, EMA, RSI, or MACD crosses below)
                sell_price = row['Close']
                execute_sell_order(
                    symbol,
                    holdings,
                    sell_price,
                    initial_balance,
                    final_balance,
                    row['SMA'],
                    row['EMA'],
                    rsi.iloc[-1],
                    macd.iloc[-1]
                )
                holdings = 0
                sell_points.append((row['Time'], sell_price))

            print("[+] Sleeping...")
            time.sleep(10)  # Sleep for 15 minutes for the next interval

        # Calculate the final balance after all transactions
        final_balance = calculate_profit_loss(initial_balance)
        print("Final Balance:", final_balance)

        # Plot the price chart with buy and sell points
        plt.figure(figsize=(12, 6))
        plt.plot(df['Time'], df['Close'], label='Price', color='blue')
        buy_times, buy_prices = zip(*buy_points)
        sell_times, sell_prices = zip(*sell_points)
        plt.scatter(buy_times, buy_prices, marker='^', color='green', label='Buy Signal')
        plt.scatter(sell_times, sell_prices, marker='v', color='red', label='Sell Signal')
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.savefig('chart.png')
        plt.show()
        
    except Exception as e:
        print("Error in backtesting:", e)

if __name__ == "__main__":
    symbol = 'BTCUSDT'
    sma_window = 21
    ema_window = 9
    buy_threshold = 0.01
    sell_threshold = -0.01
    interval = Client.KLINE_INTERVAL_1HOUR
    initial_balance = 350

    while True:
        try:
            backtest_sma_ema_strategy(symbol, sma_window, ema_window, buy_threshold, sell_threshold, interval, initial_balance)
            user_input = input("Press Enter to continue or 'q' to quit: ")
            if user_input.strip().lower() == 'q':
                break
        except Exception as e:
            print("Error in main loop:", e)
            time.sleep(30)
