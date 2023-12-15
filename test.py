from binance.client import Client
import time

# Binance API keys for the test environment
api_key = 'tTXlgT7hbCmQYf5c2tDkgKSXn7HfSXrd3uxOh9H5dxWcuzbUKLga2Adyd6oyUPeq'
api_secret = 'U8pbEJvyUwfmNuWGtnlrW2BN0r4TCuuOAGLQwFFt7VzNXtJNCt1WCKkXH9vYEi07'

# Create a Binance test client
client = Client(api_key, api_secret, testnet=True)

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
        print("Order ID:", order['orderId'])
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
        print("Order ID:", order['orderId'])
    except Exception as e:
        print("Error executing sell order:", e)

def main():
    # Define the trading pair and order parameters
    symbol = 'BTCUSDT'  # Note: Use 'BTCUSDT' instead of 'BTC/USDT'
    buy_quantity = 0.01
    buy_price = 35000  # Replace with your desired buy price
    sell_quantity = 0.01
    sell_price = 40000  # Replace with your desired sell price

    # Execute a buy order
    execute_buy_order(symbol, buy_quantity, buy_price)

    # Sleep for a while (e.g., 10 seconds) before executing a sell order
    time.sleep(10)

    # Execute a sell order
    execute_sell_order(symbol, sell_quantity, sell_price)

if __name__ == "__main__":
    main()
