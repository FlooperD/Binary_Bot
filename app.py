from flask import Flask, render_template, jsonify, request
from binance.client import Client
import datetime

app = Flask(__name__)

# Replace with your Binance API credentials
api_key = 'FEOKFzr1eTE5XkFM5q2Bdmu7k8KHbimFBnsbgYDZDY1W4qg2cpbZPgUYrKxitDin'
api_secret = 'L1yojgYtxG7HxlDiMI52KktMJ3XKgXvWq06WG1R8SJYSzMxcHRrHQLEIihVSd19P'

# Initialize Binance client
client = Client(api_key, api_secret)

# Global variables to store the data and zoom level
timestamps = []
close_prices = []
zoom_level = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/fetch_data')
def fetch_data():
    global timestamps, close_prices, zoom_level

    # Fetch the zoom level from the query parameters
    zoom = request.args.get('zoom')

    # If a zoom level is provided, update the global zoom_level variable
    if zoom:
        zoom_level = int(zoom)

    # Fetch the latest kline data for BTC/ETH
    symbol = 'BTCUSDT'
    klines = client.get_klines(symbol=symbol, interval=Client.KLINE_INTERVAL_1MINUTE, limit=1)

    # Extract the timestamp and close price from the latest kline
    timestamp = datetime.datetime.fromtimestamp(int(klines[0][0]) / 1000)
    close_price = float(klines[0][4])

    # Append the data to the global variables
    timestamps.append(timestamp)
    close_prices.append(close_price)

    # Trim the data to only keep the last 100 points
    timestamps = timestamps[-100:]
    close_prices = close_prices[-100:]

    # Apply the zoom level if available
    if zoom_level is not None:
        timestamps = timestamps[-zoom_level:]
        close_prices = close_prices[-zoom_level:]

    # Create a dictionary with the updated data
    data = {
        'timestamps': [ts.strftime('%Y-%m-%d %H:%M:%S') for ts in timestamps],
        'close_prices': close_prices
    }

    # Return the data as JSON response
    return jsonify(data)

if __name__ == '__main__':
    app.run(debug=True)
