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
import matplotlib.pyplot as plt
import os
import pickle
import tensorflow as tf
import joblib
from sklearn.model_selection import TimeSeriesSplit
from tqdm import tqdm
from tqdm.keras import TqdmCallback
from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, LSTM, Dropout
from keras_tuner import HyperModel
from keras_tuner.tuners import RandomSearch
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from binance.client import Client
from telegram import Update
from telegram.ext import Updater, CommandHandler, CallbackContext
from pmdarima.arima import auto_arima
from pandarallel import pandarallel

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning, module='statsmodels')
warnings.filterwarnings("ignore")

#Binance API keys
api_key = 'tu41sALIg5JpPX1Ejbg8tPDvZugVrNFer8F5VMkkSaowZxlg0Y5AW1bNA1j5Ym09'
api_secret = 'xcndFpLdgWffXIpLVVB8QkkKV76ua7cVJjsmBYO9p7MZrOJxA5gIJl519ka9kmKl'

#Telegram API
API_TOKEN = '5500076996:AAHC1JSDHgJVDhDqTH6LbIJt783gGZM2HSs'
CHAT_ID = '5044502086'
message = "Trading started"

csv_file_path = 'historical_data.csv'

client = Client(api_key, api_secret)

current_position = None

tf.config.threading.set_inter_op_parallelism_threads(24)
tf.config.threading.set_intra_op_parallelism_threads(24)

gpus = tf.config.list_physical_devices('GPU')
tf.config.set_visible_devices(gpus[:1], 'GPU')

pandarallel.initialize(progress_bar=True, nb_workers=24)

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

def get_min_lot_size(symbol_info):
    filters = symbol_info['filters']
    min_qty = 0
    min_notional_value = 0
    for filter in filters:
        if filter['filterType'] == 'LOT_SIZE':
            min_qty = float(filter['minQty'])
        if filter['filterType'] == 'MIN_NOTIONAL':
            min_notional_value = float(filter['minNotional'])
            print(f"Min notional value: {min_notional_value}")
    return min_qty, min_notional_value
class TqdmKerasTunerCallback(TqdmCallback):
    def on_epoch_end(self, epoch, logs=None):
        super().on_epoch_end(epoch, logs)
        self.tqdm.desc = f"Epoch {epoch + 1}/{self.params['epochs']}"
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
        'enableRateLimit': True,
        'options': {
              'adjustForTimeDifference': True,  # Automatically adjust the request timestamp
    },
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

def fetch_wallet_balance(currency):
    try:
        balance = binance.fetch_balance()
        return balance['free'].get(currency, 0)
    except Exception as e:
        print(f"Error fetching balance: {e}")
        return None

def fetch_historical_data(symbol, timeframe, since=None, max_datapoints=2000, limit=1000):
    ohlcv_data = []
    progress_bar = tqdm(desc="Fetching data", unit=" data points")
    while True:
        try:
            if since is not None:
                new_data = binance.fetch_ohlcv(symbol, timeframe, limit=limit, params={'startTime': since})
            else:
                new_data = binance.fetch_ohlcv(symbol, timeframe, limit=limit)
            if not new_data:
                break

            ohlcv_data.extend(new_data)
            progress_bar.update(len(new_data))

            if len(ohlcv_data) >= max_datapoints:
                break

            if len(new_data) >= limit:
                time.sleep(binance.rateLimit / 1000)

            if since is not None:
                since = int(ohlcv_data[-1][0]) + 1
            else:
                since = new_data[-1][0]

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

def get_last_window(close_prices, window_size):
    return close_prices[-window_size:]

class LSTMHyperModel(HyperModel):
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build(self, hp):
        model = Sequential()
        model.add(LSTM(units=hp.Int('units', min_value=32, max_value=256, step=50),
                       activation='relu',
                       input_shape=self.input_shape))
        model.add(Dropout(rate=hp.Float('dropout', min_value=0.1, max_value=0.5, step=0.1)))
        model.add(Dense(1))
        model.compile(optimizer=hp.Choice('optimizer', ['adam', 'rmsprop']),
                      loss='mean_squared_error')
        return model

def build_lstm_model(input_shape, units, dropout, optimizer):
    model = Sequential()
    model.add(LSTM(units, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(dropout))
    model.add(LSTM(units, return_sequences=False))
    model.add(Dropout(dropout))
    model.add(Dense(1))
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model

def train_lstm_model(close_prices, hyperparameters=None):
    window_size = 60
    X, y, scaler = prepare_data_for_lstm(close_prices, window_size)

    # Create a ModelCheckpoint callback to save the model after each epoch
    checkpoint_filepath = 'lstm_model_checkpoint.h5'
    model_checkpoint_callback = ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=False,
        monitor='val_loss',
        mode='min',
        save_best_only=True,
        verbose=1
    )

    # Split the data into training and testing sets
    train_ratio = 0.8
    train_size = int(len(X) * train_ratio)
    X_train, y_train = X[:train_size], y[:train_size]
    X_test, y_test = X[train_size:], y[train_size:]

    input_shape = (X_train.shape[1], X_train.shape[2])

    if hyperparameters is not None:
        lstm_model = build_lstm_model(input_shape, hyperparameters['units'], hyperparameters['dropout'], hyperparameters['optimizer'])
        epochs = hyperparameters['epochs']
        batch_size = hyperparameters['batch_size']
    else:
        lstm_model = build_lstm_model(input_shape, 50, 0.2, Adam(learning_rate=0.001))
        epochs = 50
        batch_size = 1

    lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    # Load weights from checkpoint file if it exists
    if os.path.exists(checkpoint_filepath):
        print("Loading model from checkpoint file.")
        lstm_model = load_model(checkpoint_filepath)
    else:
        # ... (continue with the training process)
        lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        history = lstm_model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=[model_checkpoint_callback],
            verbose=1
        )

        # Load the best model from the checkpoint file
        if os.path.exists(checkpoint_filepath):
            best_lstm_model = load_model(checkpoint_filepath)
        else:
            print("Warning: No best model found. Check the search process.")
            return None, None

        # Save the final model and scaler
        best_lstm_model.save("lstm_model.h5")

    return lstm_model, scaler

def train_model_arima(data):
    # Train ARIMA model
    model = pm.auto_arima(data,
                          seasonal=False,
                          suppress_warnings=True,
                          stepwise=True)
    
    # Save the trained ARIMA model
    joblib.dump(model, 'arima_model.pkl')
    
    return model

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=input_shape, return_sequences=True))
    model.add(LSTM(50, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def save_and_send_graph_image(close_prices, arima_predictions, lstm_predictions, ensemble_predictions):
    plt.figure(figsize=(14, 7))
    plt.plot(close_prices, label="Actual Prices", color='blue')
    plt.axvline(x=len(close_prices) - 1, color='green', linestyle='--')

    # Combine historical and future predictions
    arima_predictions = np.append(close_prices[:-1], arima_predictions)
    lstm_predictions = np.append(close_prices[:-1], lstm_predictions)
    ensemble_predictions = np.append(close_prices[:-1], ensemble_predictions)

    plt.plot(arima_predictions, label="ARIMA Predictions", color='red', linestyle='--')
    plt.plot(lstm_predictions, label="LSTM Predictions", color='orange', linestyle='--')
    plt.plot(ensemble_predictions, label="Ensemble Predictions", color='purple', linestyle='--')

    plt.xlabel("Datapoints")
    plt.ylabel("Price")
    plt.legend()

    img_path = "predictions.png"
    plt.savefig(img_path)
    plt.show()
    plt.close()

    # Send the graph image via Telegram
    # send_telegram_image(img_path)

    # Print the data and send it via Telegram
    data_text = f"""Actual Prices: {close_prices}
ARIMA Prediction: {arima_predictions[-1]}
LSTM Prediction: {lstm_predictions[-1]}
Ensemble Prediction: {ensemble_predictions[-1]}
"""
    print(data_text)
    send_telegram_message(data_text)

def send_telegram_image(image_path):
    url = f"https://api.telegram.org/bot{API_TOKEN}/sendPhoto"
    with open(image_path, 'rb') as image:
        files = {'photo': image}
        data = {'chat_id': CHAT_ID}
        response = requests.post(url, files=files, data=data)

    if response.status_code == 200:
        print("Telegram image sent successfully.")
    else:
        print(f"Error sending Telegram image. Status code: {response.status_code}")

def prepare_data_for_lstm(data, window_size):
    # data = data.values.reshape(-1, 1)  # Remove this line
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data.reshape(-1, 1))

    X, y = [], []
    for i in range(window_size, len(data_scaled)):
        X.append(data_scaled[i - window_size:i, 0])
        y.append(data_scaled[i, 0])

    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    return X, y, scaler

def load_lstm_model_and_scaler_and_model_arima():
    lstm_model = load_model("lstm_model.h5")

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("model_arima.pkl", "rb") as pkl_file:
        model_arima = pickle.load(pkl_file)

    return lstm_model, scaler, model_arima

def create_dataset(data, window_size):
    x, y = [], []
    for i in range(len(data) - window_size - 1):
        x.append(data[i:(i + window_size), 0])
        y.append(data[i + window_size, 0])
    return np.array(x), np.array(y)

def tune_lstm_model(series):
    # Define the hyperparameter search space
    units_space = [50, 100, 150, 200]
    dropout_space = [0.1, 0.2, 0.3, 0.4, 0.5]
    optimizer_space = ['adam', 'rmsprop']

    best_score = float('inf')
    best_hyperparameters = None

    for units, dropout, optimizer in itertools.product(units_space, dropout_space, optimizer_space):
        model = Sequential()
        model.add(LSTM(units=units, activation='relu', input_shape=(60, 1)))
        model.add(Dropout(dropout))
        model.add(Dense(1))

        model.compile(optimizer=optimizer, loss='mse')

        # Perform time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        mse_scores = []

        for train_index, test_index in tscv.split(series):
            train, test = series[train_index], series[test_index]

            # Scaling the data
            scaler = MinMaxScaler(feature_range=(0, 1))
            train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
            test_scaled = scaler.transform(test.values.reshape(-1, 1))

            # Prepare data for LSTM
            X_train, y_train = create_dataset(train_scaled, 60)
            X_test, y_test = create_dataset(test_scaled, 60)

            X_train = np.array(X_train)
            X_test = np.array(X_test)

            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

            model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)
            mse_scores.append(mse)

        mean_mse = np.mean(mse_scores)

        if mean_mse < best_score:
            best_score = mean_mse
            best_hyperparameters = {'units': units, 'dropout': dropout, 'optimizer': optimizer}

    return best_score, best_hyperparameters

def find_best_arima_order(series):
    best_aic = np.inf
    best_order = None
    best_model = None

    # Define the p, d, and q parameters to take any value between 0 and 2
    p = d = q = range(0, 3)

    # Generate all different combinations of p, q, and q triplets
    pdq = list(itertools.product(p, d, q))

    for order in pdq:
        try:
            model_arima = ARIMA(series, order=order)
            model_fit_arima = model_arima.fit(disp=0)
            aic = model_fit_arima.aic

            if aic < best_aic:
                best_aic = aic
                best_order = order
                best_model = model_fit_arima
        except:
            continue

    return best_order, best_model

def print_metrics(y_true, y_pred, label):
    print(f"\n{label} Metrics:")
    print(f"Mean Absolute Error: {mean_absolute_error(y_true, y_pred)}")
    print(f"Mean Squared Error: {mean_squared_error(y_true, y_pred)}")
    print(f"R2 Score: {r2_score(y_true, y_pred)}")

def strategy(data, lstm_model, scaler, last_window, timestamp, last_trained_date, model_arima, window_size):
    current_position = "NONE"

    # Retrieve the current close price
    current_close = data['close'].iloc[-1]

    # Make predictions using the ARIMA and LSTM models
    y_pred_arima = model_arima.predict(n_periods=2)
    y_pred_lstm = predict_with_lstm(lstm_model, scaler, last_window, window_size, steps_ahead=2)

    # Calculate the ensemble prediction for the current timestamp
    # ensemble_prediction_current = (y_pred_arima[0] + y_pred_lstm[0]) / 2
    # ensemble_prediction_next = (y_pred_arima[1] + y_pred_lstm[1]) / 2

    ensemble_prediction_current = (y_pred_arima[0] * 0.6) + (y_pred_lstm[0] * 0.4)
    ensemble_prediction_next = (y_pred_arima[1] * 0.6) + (y_pred_lstm[1] * 0.4)

    print(f"ARIMA Prediction: {y_pred_arima[0]}")
    print(f"LSTM Prediction: {y_pred_lstm[0]}")
    print(f"Ensemble Prediction: {ensemble_prediction_current}")
    # Generate trading signals based on the ensemble prediction
    signal = "HOLD"
    if ensemble_prediction_next > current_close * 1: # Default (Strict) = 1.005
        signal = "BUY"
        print("[+] Buy")
    elif ensemble_prediction_next < current_close * 1: # Default (Strict) = 0.995
        signal = "SELL"
        print("[+] Sell")

    print(f"ARIMA prediction: {y_pred_arima[1]}")
    print(f"LSTM prediction: {y_pred_lstm[1]}")
    print(f"Ensemble prediction current: {ensemble_prediction_current}")
    print(f"Ensemble prediction next: {ensemble_prediction_next}")

    # send_telegram_message(f"ARIMA prediction: {y_pred_arima[1]}")
    # send_telegram_message(f"LSTM prediction: {y_pred_lstm[1]}")
    # send_telegram_message(f"Ensemble prediction current: {ensemble_prediction_current}")
    # send_telegram_message(f"Ensemble prediction next: {ensemble_prediction_next}")

    # Update the models if necessary
    if last_trained_date is None or (timestamp - last_trained_date).days >= 1:
        print("Updating models...")
        close_prices = data['close'].values
        lstm_model, scaler = train_lstm_model(close_prices)
        model_arima = train_model_arima(close_prices)
        last_trained_date = timestamp

    return signal, y_pred_arima, y_pred_lstm, ensemble_prediction_current, last_trained_date

def predict_with_arima(model_arima, data, window_size):
    arima_predictions = model_arima.predict(n_periods=window_size)
    return arima_predictions

def predict_with_lstm(lstm_model, scaler, last_window, window_size, steps_ahead=2):
    if last_window is None:
        raise ValueError("last_window cannot be None")

    y_pred_lstm = []
    for _ in range(steps_ahead):
        # Prepare the data for the LSTM model
        last_window_scaled = scaler.transform(last_window.reshape(-1, 1))
        X_test = last_window_scaled[-window_size:].reshape(1, window_size, 1)

        # Make a prediction
        y_pred = lstm_model.predict(X_test)

        # Invert the scaling
        y_pred = scaler.inverse_transform(y_pred)

        y_pred_lstm.append(y_pred[0, 0])
        last_window = np.append(last_window, y_pred[0, 0])

    return np.array(y_pred_lstm)

def lstm_predict_future(model, scaler, data, window_size):
    data_scaled = scaler.transform(data.reshape(-1, 1))
    data_scaled = data_scaled[-window_size:]
    data_scaled = np.reshape(data_scaled, (1, window_size, 1))

    lstm_predictions = model.predict(data_scaled)
    lstm_prediction_current = scaler.inverse_transform(lstm_predictions)[0][0]
    
    # Add the current prediction to the input data and predict the future step
    data_scaled = np.append(data_scaled[0], lstm_predictions)
    data_scaled = data_scaled[-window_size:]
    data_scaled = np.reshape(data_scaled, (1, window_size, 1))

    lstm_predictions_future = model.predict(data_scaled)
    lstm_prediction_future = scaler.inverse_transform(lstm_predictions_future)[0][0]

    return lstm_prediction_current, lstm_prediction_future

def calculate_ensemble_predictions(arima_predictions, lstm_predictions):
    ensemble_predictions = (arima_predictions + lstm_predictions) / 2
    return ensemble_predictions

def save_data_to_csv(data, file_path):
    if os.path.exists(file_path):
        data.to_csv(file_path, mode='a', header=False, index=False)
    else:
        data.to_csv(file_path, index=False)

def load_data_from_csv(file_path):
    return pd.read_csv(file_path)

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
    trading_currency = base_currency if current_position == "NONE" else quote_currency
    balance = fetch_wallet_balance(trading_currency)

    if balance is None:
        print("Error: Unable to fetch wallet balance.")
        return

    print(f"[+]Current position: {current_position}")
    print(f"[+]Signal: {signal}")
    print(f"{trading_currency} balance: {balance}")

    symbol = f"{base_currency}{quote_currency}"
    symbol_info = client.get_symbol_info(symbol)
    min_qty, min_notional_value = get_min_lot_size(symbol_info)

    if min_qty is None:
        print("[+]Error: Unable to fetch lot size.")
        return

    precision = abs(decimal.Decimal(str(min_qty)).as_tuple().exponent)

    if signal == "BUY" and current_position == "NONE":
        amount /= fetch_current_price(base_currency, quote_currency)
        amount = math.floor(amount / min_qty) * min_qty
        amount = round(amount, precision)
        notional_value = amount * fetch_current_price(base_currency, quote_currency)

        print("[+]Buying...")
        if notional_value >= min_notional_value:
            try:
                binance.create_market_buy_order(f"{base_currency}/{quote_currency}", amount)
                current_position = "LONG"
                print("Bought", base_currency, "using", quote_currency)
                send_telegram_message(f"Bought {amount} of {base_currency} using {quote_currency}")
            except Exception as e:
                print("Error buying:", e)
        else:
            print("[+]Trade amount does not meet the minimum notional value.")

    elif signal == "SELL" and current_position == "LONG":
        amount = balance[base_currency]['free']
        amount = math.floor(amount / min_qty) * min_qty
        amount = round(amount, precision)

        print("[+]Selling...")
        try:
            binance.create_market_sell_order(f"{base_currency}/{quote_currency}", amount)
            current_position = "NONE"
            print("Sold", base_currency, "for", quote_currency)
            send_telegram_message(f"Sold {amount} of {base_currency} for {quote_currency}")
        except Exception as e:
            print("Error selling:", e)
    else:
        print("No valid signal or position.")

def main(base_currency, quote_currency):
    models_loaded = False
    interval = '5m'
    symbol = f"{base_currency}/{quote_currency}"
    data = pd.DataFrame()
    window_size = 60  # Define the window size for the LSTM model
    min_datapoints = 5000  # Minimum number of data points required to train the models
    last_trained_date = None

    print("Fetching wallet balances:")
    base_balance = fetch_wallet_balance(base_currency)
    quote_balance = fetch_wallet_balance(quote_currency)

    if base_balance is not None:
        print(f"Available {base_currency} balance: {base_balance}")
    if quote_balance is not None:
        print(f"Available {quote_currency} balance: {quote_balance}")

    trading_currency = input(f"Enter the currency you want to allocate funds to (either {base_currency} or {quote_currency}): ").upper()
    while trading_currency not in [base_currency, quote_currency]:
        print("Invalid input. Please enter the correct currency.")
        trading_currency = input(f"Enter the currency you want to allocate funds to (either {base_currency} or {quote_currency}): ").upper()

    trading_balance = fetch_wallet_balance(trading_currency)
    trade_ratio = float(input(f"Enter the portion of your {trading_currency} balance to use (0 to 1): "))
    last_window = None
    current_position = "NONE"  # Add this line to define the initial position

    while True:
        # Fetch historical data
        historical_data = fetch_historical_data(symbol, interval)
        print("Historical Data:", historical_data)
        if data.empty:
            data = historical_data
        else:
            data = data.append(historical_data, ignore_index=True)
            print("Appended Data:", data)
            print("Fetched Data:")
            print(historical_data)
            print("\nAppended Data:")
            print(data)
            data.to_csv("historical_data.csv", mode='a', index=False)

        csv_file = "verify_historical_data.csv"
        header = not os.path.exists(csv_file)  # Add header if file doesn't exist
        data.to_csv(csv_file, mode='a', index=False, header=header)

        # Extract close prices from historical data
        close_prices = data['close'].values

        if len(data) >= min_datapoints:
            if not models_loaded:
                print("Training models...")
                lstm_model, scaler = train_lstm_model(close_prices)
                model_arima = train_model_arima(close_prices)
                models_loaded = True

            if last_window is None:
                last_window = close_prices[-window_size:]
            else:
                last_window = np.append(last_window, close_prices[-1])
                last_window = last_window[-window_size:]

            timestamp = data['timestamp'].iloc[-1]
            print("Retraining models...")
            model_arima = train_model_arima(close_prices)
            signal, y_pred_arima, y_pred_lstm, ensemble_predictions, last_trained_date = strategy(
                data, lstm_model, scaler, last_window, timestamp, last_trained_date, model_arima, window_size)

            # Save and send the graph image
            save_and_send_graph_image(close_prices, y_pred_arima, y_pred_lstm, ensemble_predictions)

            # Trading logic
            if signal == 'BUY' and current_position == "NONE":
                amount_to_trade = trading_balance * trade_ratio
                execute_trade('BUY', base_currency, quote_currency, amount_to_trade)
                print(f"Executing BUY trade for {amount_to_trade} {trading_currency}")
                current_position = "LONG"  # Update the current position

            elif signal == 'SELL' and current_position == "LONG":
                amount_to_trade = trading_balance * trade_ratio
                execute_trade('SELL', base_currency, quote_currency, amount_to_trade)
                print(f"Executing SELL trade for {amount_to_trade} {trading_currency}")
                current_position = "NONE"  # Update the current position

            else:  # HOLD
                print(f"Holding position. No trade executed.")
                
        print("\n[+] Sleeping...")
        time.sleep(300)

if __name__ == "__main__":
    custom_base_currency = input("Enter the base currency (e.g. BTC): ").upper()
    custom_quote_currency = input("Enter the quote currency (e.g. USDT): ").upper()
    main(custom_base_currency, custom_quote_currency)