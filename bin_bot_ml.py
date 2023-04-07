import ccxt
import pandas as pd
import time
import sys
from hmmlearn import hmm
import numpy as np

# Your Binance API keys
api_key = 'API Key'
api_secret = 'API Secret'

current_position = None

# Initialize the Binance API client
binance = ccxt.binance({
    'apiKey': api_key,
    'secret': api_secret,
    'enableRateLimit': True
})

def get_min_notional(symbol):
    markets = binance.load_markets()
    market = markets[symbol]
    min_notional = market['limits']['cost']['min']
    return min_notional

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

    # Fetch historical data
    symbol = 'ETH/ZAR'
    timeframe = '5m'

    # Get the minimum notational value for the selected symbol
    min_notional = get_min_notional(symbol)

    # Prompt the user for the portion of funds to allocate
    amount = float(input("How much do you want to use: "))

    # Validate the input and check if the allocated amount is greater than the minimum notational value
    while True:
        if 0 <= amount <= 2000 and amount >= min_notional:
            print(f"Allocating R{amount} of available funds for trading.")
            break
        else:
            print(f"Invalid input or the allocated amount is less than the minimum notational value of R{min_notional}.")
            amount = float(input("Please enter a new amount to use: "))
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

# Get the minimum notational value for the selected symbol
min_notional = get_min_notional(symbol)

# Print the minimum notational value for the symbol
print(f"Minimum notational value for {symbol}: R{min_notional}")

# Prompt the user for the portion of funds to allocate
amount = float(input("How much do you want to use: "))

# Validate the input and check if the allocated amount is greater than the minimum notational value
while True:
    if 0 <= amount <= 2000 and amount >= min_notional:
        print(f"Allocating R{amount} of available funds for trading.")
        break
    else:
        print(f"Invalid input or the allocated amount is less than the minimum notational value of R{min_notional}.")
        amount = float(input("Please enter a new amount to use: "))


def train_hmm_model(data, n_components=4, n_iter=1000):
    model = hmm.GaussianHMM(n_components=n_components, covariance_type="diag", n_iter=n_iter)
    model.fit(data)
    return model

def predict_hidden_states(model, data):
    hidden_states = model.predict(data)
    return hidden_states

def strategy(data):
    # Prepare the data (you can use price, returns, or any other feature)
    returns = np.log(data['close'] / data['close'].shift(1)).dropna().values.reshape(-1, 1)

    # Train the HMM model
    model = train_hmm_model(returns)

    # Predict the hidden states
    hidden_states = predict_hidden_states(model, returns)

    # Generate trading signals based on hidden states (e.g., buy when the hidden state is 1, sell when it's 0)
    signal = (hidden_states == 1)
    
    # Create a new DataFrame with the signal column
    result = data.iloc[1:].copy()  # Drop the first row as it has NaN in returns
    result['signal'] = signal
    print(result) # Print the result dataframe
    return result

# Implement your real-time trading logic here
def trade_logic():
    global data, iteration_counter, result

    iteration_counter = 0
    update_interval = 10 # Update the model every 10 iterations

    retries = 3
    retry_delay = 30  # 60 seconds

    # Update the model every 'update_interval' iterations
    if iteration_counter % update_interval == 0:
        print("Updating the strategy...")
        result = strategy(data)
        new_signal = result['signal'].iloc[-1]
    else:
        new_signal = result['signal'].iloc[-1]
        new_signal = not new_signal # Flip the signal

    iteration_counter += 1

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

    # Update the main data DataFrame with new data
  
    data = pd.concat([data, new_data], ignore_index=True)

    # Apply the strategy on the updated data
    result = strategy(data)
    new_signal = result['signal'].iloc[-1]

    # Get the current available balance
    wallet_balance = fetch_wallet_balance()
    if wallet_balance is None:
        print("Unable to fetch wallet balance. Exiting the program.")
        sys.exit()

    available_zar_balance = wallet_balance['free']['ZAR']
    available_eth_balance = wallet_balance['free']['ETH']

    if new_signal:
        # Buy logic
        if available_zar_balance >= amount:
            try:
                order = binance.create_market_buy_order(symbol, amount / result['close'].iloc[-1])
                print("Buy order placed:", order)
            except Exception as e:
                print("Error placing buy order:", e)
        else:
            print("Not enough balance to place buy order")
    else:
        # Sell logic
        if available_eth_balance > 0:
            try:
                order = binance.create_market_sell_order(symbol, available_eth_balance)
                print("Sell order placed:", order)
            except Exception as e:
                print("Error placing sell order:", e)
        else:
            print(f"No {symbol} available to sell")
    
    print(new_signal)
    print(data)

data = fetch_historical_data(symbol, timeframe)

# Run the bot in a loop
while True:
    trade_logic()
    time.sleep(60)  
