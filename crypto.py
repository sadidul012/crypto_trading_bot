import pandas as pd
from metrics import stochastics, rsi


assets_list = ['BTCUSDT', 'ETHUSDT', "TRX"]
assets = [coin.lower() + '@kline_1m' for coin in assets_list]
assets = "/".join(assets)

coins = dict(zip(assets_list, [None] * len(assets_list)))
metrics = dict(zip(assets_list, [None] * len(assets_list)))
columns = ['s', "d", 'o', 'c', 'h', 'l', 'v', 'n']
position = "sell"
count = 0
price = 0
coin_count = 0
profits = 0
loss = 0
loss_count = 0


def manipulation(source):
    global position, count, price, coin_count, profits, loss_count, loss
    source = dict(zip(columns, source))
    df = pd.DataFrame([source])
    coins[source['s']] = pd.concat([coins[source['s']].iloc[-17:], df]) if coins[source['s']] is not None else df
    df = coins[source['s']]
    df = rsi(df, "c", 10)
    df = stochastics(df, 'l', 'h', 'c', 14, 3)

    metrics[source['s']] = pd.concat([metrics[source['s']].iloc[-17:], df.iloc[-1:]]) if metrics[source['s']] is not None else df
    if df.shape[0] >= 18:
        data = df.iloc[-1].to_dict()
        if position == "sell" and (data["rsi14"] < 60 and data["k_slow"] < 20 and data["d_slow"] < 20):
            position = "buy"
            price = data["c"]
            count += 1
            coin_count = 100/data["c"]
            print(position, count, data["d"], data["c"])
            print("coins {:2f}".format(coin_count))

        if position == "buy" and (data["rsi14"] > 60 and data["k_slow"] > 80 and data["d_slow"] > 80):
            position = "sell"
            profit = (data["c"] - price) * coin_count
            profits += profit
            if profit <= 0:
                loss += profit
                loss_count += 1
            print(position, count, data["d"], data["c"])
            print("profit {:2f}".format(profit))
            print()


def main():
    df = pd.read_csv("data/crypto/coin_Tron.csv")
    old_columns = ["Symbol", "Date", "High", "Low", "Open", "Close", "Volume", "Marketcap"]
    new_columns = ["s", "d", "o", "c", "h", "l", "v", "n"]
    show_columns = ["d", "rsi14", "k_slow", "d_slow"]

    df = df[old_columns]
    df.columns = new_columns

    # df = df.iloc[-22:]
    for i, row in df.iterrows():
        manipulation(list(row.values))

    print("cumulative profit", profits)
    print("cumulative loss", loss)
    print("loss count", loss_count)
    # df = rsi(df, "c", 10)
    # df = stochastics(df, 'l', 'h', 'c', 14, 3)
    # print(df[show_columns].to_string())
    # slow_k = stochs['k_slow'].values
    # fast_k = stochs['k_fast'].values


if __name__ == '__main__':
    main()
