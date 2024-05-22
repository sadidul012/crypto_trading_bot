import json
import websocket
from crypto import manipulation, columns, assets_list

assets = [coin.lower() + '@kline_1m' for coin in assets_list]
assets = "/".join(assets)

count = 0


def on_message(s, message):
    global count
    data = json.loads(message)
    # manipulation(data['data']['s'], [[data['data']['k'][k] for k in columns]][0])
    count += 1
    print(count)


if __name__ == '__main__':

    socket = "wss://stream.binance.com:9443/stream?streams=" + assets
    ws = websocket.WebSocketApp(socket, on_message=on_message)
    ws.run_forever()
