import ccxt
import pandas as pd
import pandas_ta as pta
import time
import numpy as np
import requests
import warnings
import json
import decimal
import math
import itertools
import pmdarima as pm
from tqdm import tqdm
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')

#Binance API keys
api_key = 'Your API Key here'
api_secret = 'Your API_Secret here'

#Telegram API
API_TOKEN = 'Your API Key here'
CHAT_ID = 'Your chat ID here'
message = "Trading started"

current_position = None

def create_features(df):
    # Calculate moving averages
    df['ma7'] = df['close'].rolling(window=7).mean()
    df['ma25'] = df['close'].rolling(window=25).mean()

    # Calculate RSI
    df['rsi'] = pta.rsi(df['close'], length=14)

    # Calculate MACD
    macd_df = pta.macd(df['close'], fast=12, slow=26, signal=9)

    # Assign the MACD signal line to the main DataFrame
    df['macd_signal'] = macd_df['MACDs_12_26_9']

    return df

def debug_print_request_params(params, url):
    print("Request parameters:")
    print(json.dumps(params, indent=2))
    print("Request URL:")
    print(url)

def get_min_lot_size(base_currency, quote_currency):
    try:
        markets = binance.load_markets()
        symbol = f"{base_currency}/{quote_currency}"
        market = markets[symbol]
        filters = market['info']['filters']
        for filter in filters:
            if filter['filterType'] == 'LOT_SIZE':
                return float(filter['stepSize'])
    except Exception as e:
        print("Error fetching lot size:", e)
        return None
class DebuggableBinance(ccxt.binance):
    def request(self, path, api='public', method='GET', params={}, headers=None, body=None, config={}, context={}):
        url = self.implode_params(self.urls['api'][api] + path, params)
        debug_print_request_params(params, url)
        return super(DebuggableBinance, self).request(path, api, method, params, headers, body, config, context)

binance = DebuggableBinance({
    'apiKey': 'YOUR_API_KEY',
    'secret': 'YOUR_SECRET_KEY',
})

def initialize_binance_client(api_key, api_secret):
    exchange = ccxt.binance({
        'apiKey': api_key,
        'secret': api_secret,
        'enableRateLimit': True
    })
    return exchange

binance = initialize_binance_client(api_key, api_secret)

def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{API_TOKEN}/sendMessage?chat_id={CHAT_ID}&text={message}"
    response = requests.get(url)
    if response.status_code == 200:
        print("Telegram message sent successfully.")
    else:
        print(f"Error sending Telegram message. Status code: {response.status_code}")

def fetch_wallet_balance():
    try:
        balance = binance.fetch_balance()
        return balance
    except Exception as e:
        print(f"Error fetching balance: {e}")
        return None

wallet_balance = fetch_wallet_balance()

if wallet_balance:
    # Print the available funds
    print(f"Available wallet balance:")
    for currency, balance in wallet_balance['free'].items():
        if balance > 0:
            print(f"{currency}: {balance}")

    # Prompt the user for the portion of funds to allocate
    # amount = float(input("How much do you want to use: "))

    retry_delay = 60
    retries = 3

def fetch_historical_data(symbol, timeframe, since=None, max_datapoints=5000):
    
    ohlcv_data = []
    progress_bar = tqdm(desc="Fetching data", unit=" data points")
    counter = 0
    limit = min(2000, max_datapoints)  # Update the limit to the minimum of 1000 and max_datapoints

    while True:
        try:
            if counter == 0:
                if since is not None:
                    new_data = binance.fetch_ohlcv(symbol, timeframe, limit=limit, params={'startTime': since})
                else:
                    new_data = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
            else:
                new_data = binance.fetch_ohlcv(symbol, timeframe, limit=limit, params={'startTime': int(ohlcv_data[-1][0]) + 1})
            if not new_data:
                break

            ohlcv_data.extend(new_data)
            progress_bar.update(len(new_data))
            counter += len(new_data)

            if counter >= max_datapoints:
                break

            time.sleep(binance.rateLimit / 1000)
        except Exception as e:
            print(f"Error fetching data: {e}")
            time.sleep(10)

    progress_bar.close()

    # Create the DataFrame
    df = pd.DataFrame(ohlcv_data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

    # Convert the data types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df['open'] = df['open'].astype(float)
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['volume'] = df['volume'].astype(float)

    # Create features
    df = create_features(df)

    print(f"Fetched {len(df)} data points")

    return df
class LSTMHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        model.add(LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32),
                       activation='relu',
                       input_shape=self.input_shape))
        model.add(Dropout(rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(1))
        model.compile(optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
                      loss='mean_squared_error')
        return model

def train_arima_model(close_prices):
    model_arima = pm.auto_arima(close_prices,
                                seasonal=False,
                                stepwise=True,
                                suppress_warnings=True,
                                trace=False,
                                error_action='ignore')
    return model_arima

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def prepare_data_for_lstm(data, window_size):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))

    X, y = [], []
    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i - window_size:i, 0])
        y.append(data_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

def train_lstm_model(X_train, y_train, X_test, y_test):
    input_shape = (X_train.shape[1], X_train.shape[2])
    hypermodel = LSTMHyperModel(input_shape)

    tuner = RandomSearch(hypermodel,
                         objective='val_loss',
                         max_trials=10,
                         executions_per_trial=3,
                         directory='keras_tuner',
                         project_name='lstm_hyperparameter_tuning')

    tuner.search(X_train, y_train,
                 epochs=15,
                 batch_size=64,
                 validation_data=(X_test, y_test),
                 verbose=0)

    best_lstm_model = tuner.get_best_models(num_models=1)[0]

    return best_lstm_model

def print_metrics(y_true, y_pred, label):
    print(f"\n{label} Metrics:")
    print(f"Mean Absolute Error: {mean_absolute_error(y_true, y_pred)}")
    print(f"Mean Squared Error: {mean_squared_error(y_true, y_pred)}")
    print(f"R2 Score: {r2_score(y_true, y_pred)}")

def strategy(data, model_lstm, scaler, last_window):
    # Prepare the data (you can use price, returns, or any other feature)
    close_prices = data['close'].values

    p_values = range(0, 4)
    d_values = range(0, 3)
    q_values = range(0, 4)

    min_aic = float("inf")
    best_order = None

    for p, d, q in itertools.product(p_values, d_values, q_values):
        try:
            model_arima = ARIMA(close_prices, order=(p, d, q))
            model_fit_arima = model_arima.fit()
            aic = model_fit_arima.aic
            if aic < min_aic:
                min_aic = aic
                best_order = (p, d, q)
        except Exception as e:
            continue

    print(f"Best order: {best_order}, AIC: {min_aic}")

    # Train the ARIMA model
    model_arima = ARIMA(close_prices, order=(1, 1, 1))
    model_fit_arima = model_arima.fit()

    # Display ARIMA model metrics
    print("\nARIMA Model Metrics:")
    print(f"AIC: {model_fit_arima.aic}")

    # Predict the next price using ARIMA
    next_price_prediction_arima = model_fit_arima.forecast(steps=1)[0]

    # Predict the next price using LSTM
    input_data = np.reshape(last_window, (1, last_window.shape[0], 1))
    next_price_prediction_lstm = model_lstm.predict(input_data)
    next_price_prediction_lstm = scaler.inverse_transform(next_price_prediction_lstm)[0, 0]

    # Calculate the ensemble prediction (average of ARIMA and LSTM predictions)
    next_price_prediction = (next_price_prediction_arima + next_price_prediction_lstm) / 2

    # Generate trading signals based on the predicted price (e.g., buy if the
    # predicted price is higher than the current price, sell otherwise)
    current_price = close_prices[-1]

    if next_price_prediction > current_price:
        return 'buy'
    else:
        return 'sell'

def fetch_current_price(base_currency, quote_currency):
    try:
        symbol = f"{base_currency}/{quote_currency}"
        ticker = binance.fetch_ticker(symbol)
        return ticker['last']
    except Exception as e:
        print(f"Error fetching current price: {e}")
        return None

def execute_trade(signal, base_currency, quote_currency, amount, initial=False):
    global current_position
    balance = fetch_wallet_balance()

    if balance is None:
        print("Error: Unable to fetch wallet balance.")
        return

    step_size = get_min_lot_size(base_currency, quote_currency)
    if step_size is None:
        print("Error: Unable to fetch lot size.")
        return

    precision = abs(decimal.Decimal(str(step_size)).as_tuple().exponent)

    if signal == "BUY" and current_position == "NONE":
        amount /= fetch_current_price(base_currency, quote_currency)
        amount = math.floor(amount / step_size) * step_size
        amount = round(amount, precision)

        print("Buying...")
        try:
            binance.create_market_buy_order(f"{base_currency}/{quote_currency}", amount)
            current_position = "LONG"
            print("Bought", base_currency, "using", quote_currency)
        except Exception as e:
            print("Error buying:", e)

    elif signal == "SELL" and current_position == "LONG":
        amount = balance[base_currency]['free']
        amount = math.floor(amount / step_size) * step_size
        amount = round(amount, precision)

        print("Selling...")
        try:
            binance.create_market_sell_order(f"{base_currency}/{quote_currency}", amount)
            current_position = "NONE"
            print("Sold", base_currency, "for", quote_currency)
        except Exception as e:
            print("Error selling:", e)
    else:
        print("No valid signal or position.")

def main(base_currency, quote_currency):
    symbol = f'{base_currency}/{quote_currency}'
    timeframe = '5m'

    wallet_balance = fetch_wallet_balance()
    if wallet_balance:
        print(f"Available wallet balance for {quote_currency}: {wallet_balance['free'][quote_currency]}")

        amount = float(input(f"How much {quote_currency} do you want to use: "))

    initial_sell = wallet_balance['free'][base_currency] > 0.001

    data = fetch_historical_data(symbol, timeframe, max_datapoints=5000)

    # Print DataFrame before training
    print("\nData before training:")
    print(data.head())

    close_prices = data['close'].values
    window_size = 60
    X, y, scaler = prepare_data_for_lstm(close_prices, window_size)

    model_lstm = create_lstm_model((X.shape[1], 1))
    model_lstm.fit(X, y, epochs=20, batch_size=32, verbose=0)

    # Predict the prices using LSTM model
    y_pred_lstm = model_lstm.predict(X)
    y_pred_lstm = scaler.inverse_transform(y_pred_lstm)

    # Print DataFrame after training
    print("\nData after training:")
    data_after_training = data.copy()
    data_after_training = data_after_training.iloc[window_size:]
    data_after_training['predicted_close'] = y_pred_lstm
    print(data_after_training.head())

    # Print model metrics
    print_metrics(data['close'].iloc[window_size:].values, y_pred_lstm[:, 0], "LSTM")

    while True:
        data = fetch_historical_data(symbol, timeframe, max_datapoints=5000)

        last_window = data['close'].values[-window_size:]

        signal = strategy(data, model_lstm, scaler, last_window)

        if initial_sell and current_position is None:
            execute_trade('sell', base_currency, quote_currency, amount, initial=True)
            initial_sell = False
        else:
            execute_trade(signal, base_currency, quote_currency, amount)

        sleep_time = 300
        print(f"Sleeping for {sleep_time} seconds")
        time.sleep(sleep_time)

if __name__ == "__main__":
    custom_base_currency = input("Enter the base currency (e.g., 'BTC'): ")
    custom_quote_currency = input("Enter the quote currency (e.g., 'USDT'): ")
    main(custom_base_currency, custom_quote_currency)