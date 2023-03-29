import ccxt
import pandas as pd
import time
import sys

# Your Binance API keys
api_key = '2sP9lLF3SkkaRMZMg3tnhpH67jd9zh9ekf3pycfi2CYoPePR3xQCaGBzT4gfGpR4'
api_secret = 'p68DdveGFecovYQONQpFg0H8BlrJLghb7XevT4Ay939amlwXGZRWPJgtB2zX6xBB'

current_position = None

# Initialize the Binance API client
binance = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True
})

def fetch_wallet_balance():
    try:
        balance = binance.fetch_balance()
        return balance
    except Exception as e:
        print(f"Error fetching balance: {e}")
        return None
# Fetch wallet balance
wallet_balance = fetch_wallet_balance()
if wallet_balance:
    # Print the available funds
    print(f"Available wallet balance:")
    for currency, balance in wallet_balance['free'].items():
        if balance > 0:
            print(f"{currency}: {balance}")

    # Prompt the user for the portion of funds to allocate
    amount = float(input("How much do you want to use: "))

    retry_delay = 60
    retries = 3

    # Validate the input
    if 0 <= amount <= 2000:
        print(f"Allocating R{amount}0 of available funds for trading.")
        # You can now use the 'portion' variable in your trading logic
    else:
        print("Invalid input.")
else:
    print("Unable to fetch wallet balance.")

# Function to fetch historical data
def fetch_historical_data(symbol, timeframe, since=None):
    ohlcv_data = []
    while True:
        try:
            new_data = binance.fetch_ohlcv(symbol, timeframe, since)
            if not new_data:
                break
            since = new_data[-1][0] + 1
            ohlcv_data.extend(new_data)
            time.sleep(binance.rateLimit / 1000)
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(10)
    return pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Fetch historical data
symbol = 'ETH/ZAR'
timeframe = '5m'
data = fetch_historical_data(symbol, timeframe)
print(data)

def strategy(data):
    # Example: Simple moving average crossover strategy
    SMA_short = data['close'].rolling(window=50).mean()
    SMA_long = data['close'].rolling(window=200).mean()
    signal = (SMA_short > SMA_long)
    signal.fillna(False, inplace=True)  # Fill NaN values with False

    # Create a new DataFrame with the signal column
    result = data.copy()
    result['signal'] = signal
    print(result)
    return result

# Implement your real-time trading logic here
def trade_logic():
    retries = 3
    retry_delay = 30  # 60 seconds

    for _ in range(retries):
        # Fetch new data and apply the strategy
        new_data = fetch_historical_data(symbol, timeframe, since=data['timestamp'].iloc[-1] + 1)

        if not new_data.empty:
            break

        print("No new data fetched. Retrying in {} seconds...".format(retry_delay))
        time.sleep(retry_delay)

    if new_data.empty:
        print("No new data fetched after {} retries. Skipping this iteration.".format(retries))
        return

    new_data = strategy(new_data)
    new_signal = new_data['signal'].iloc[-1]

    print(new_data)

    # Get the current available balance
    wallet_balance = fetch_wallet_balance()
    if wallet_balance is None:
        print("Unable to fetch wallet balance. Exiting the program.")
        sys.exit()

    available_zar_balance = wallet_balance['free']['ZAR']
    available_btc_balance = wallet_balance['free']['BTC']

    if new_signal:
        # Buy logic
        if available_zar_balance >= amount:
            try:
                order = binance.create_market_buy_order(symbol, amount / new_data['close'].iloc[-1])
                print("Buy order placed:", order)
            except Exception as e:
                print("Error placing buy order:", e)
        else:
            print("Not enough balance to place buy order")
    else:
        # Sell logic
        if available_btc_balance > 0:
            try:
                order = binance.create_market_sell_order(symbol, available_btc_balance)
                print("Sell order placed:", order)
            except Exception as e:
                print("Error placing sell order:", e)
        else:
            print(f"No {symbol} available to sell")

# Run the bot in a loop
while True:
    trade_logic()
    time.sleep(60)  
