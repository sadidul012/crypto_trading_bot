import json
import websocket
from crypto import manipulation, columns, assets


def on_message(ws, message):
    data = json.loads(message)
    manipulation([[data['data']['k'][k] for k in columns]])


socket = "wss://stream.binance.com:9443/stream?streams=" + assets
ws = websocket.WebSocketApp(socket, on_message=on_message)


ws.run_forever()
