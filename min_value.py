from binance.client import Client

# Initialize the Binance client (use your API key and secret)
api_key = 'tTXlgT7hbCmQYf5c2tDkgKSXn7HfSXrd3uxOh9H5dxWcuzbUKLga2Adyd6oyUPeq'
api_secret = 'U8pbEJvyUwfmNuWGtnlrW2BN0r4TCuuOAGLQwFFt7VzNXtJNCt1WCKkXH9vYEi07'
client = Client(api_key, api_secret)

# Define the trading pair
symbol = 'BTCUSDT'

# Get the exchange info to retrieve trading pair details
exchange_info = client.get_exchange_info()

# Find the minimum order amount for BTCUSDT
for symbol_info in exchange_info['symbols']:
    if symbol_info['symbol'] == symbol:
        min_order_amount = float(symbol_info['filters'][1]['minQty'])  # Filter index 1 is for LOT_SIZE
        break

print(f"Minimum order amount for {symbol}: {min_order_amount}")
