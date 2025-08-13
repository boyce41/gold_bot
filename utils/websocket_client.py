import websocket
import json

class TwelveDataWebSocketClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.ws = None
        self.is_connected = False

    def connect(self):
        self.ws = websocket.WebSocketApp(
            f'wss://ws.twelvedata.com/v1/quotes?apikey={self.api_key}',
            on_message=self.on_message,
            on_error=self.on_error,
            on_close=self.on_close
        )
        self.ws.on_open = self.on_open
        self.ws.run_forever()

    def on_open(self):
        self.is_connected = True
        print('WebSocket connection established.')

    def on_message(self, message):
        data = json.loads(message)
        # Process the message according to your needs
        print(data)

    def on_error(self, error):
        print(f'WebSocket error: {error}')
        self.stop()

    def on_close(self):
        print('WebSocket connection closed.')
        self.is_connected = False

    def stop(self):
        if self.ws:
            self.ws.close()
        self.is_connected = False

    def validate_api_key(self):
        # Logic to validate the API key (optional)
        return len(self.api_key) > 0

    def get_connection_status(self):
        return self.is_connected
