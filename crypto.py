import pandas as pd
from metrics import stochastics, rsi


assets_list = ['BTCUSDT', 'ETHUSDT', "TRXUSDT"]
coins = dict(zip(assets_list, [None] * len(assets_list)))
metrics = dict(zip(assets_list, [None] * len(assets_list)))
columns = ["t", "o", "h", "l", "c", "v"]
position = "sell"
count = 0
price = 0
coin_count = 0
profits = 0
loss = 0
loss_count = 0
invest = 100
highest = 0


def manipulation(symbol, source):
    global position, count, price, coin_count, profits, loss_count, loss, invest, highest
    source = dict(zip(columns, source))
    df = pd.DataFrame([source])
    coins[symbol] = pd.concat([coins[symbol].iloc[-17:], df]) if coins[symbol] is not None else df
    df = coins[symbol]
    df = rsi(df, "c", 10)
    df = stochastics(df, 'l', 'h', 'c', 14, 3)

    if source["c"] > highest:
        highest = source["c"]

    metrics[symbol] = pd.concat([metrics[symbol].iloc[-17:], df.iloc[-1:]]) if metrics[symbol] is not None else df
    if df.shape[0] >= 18:
        data = df.iloc[-1].to_dict()
        date = pd.to_datetime(data["t"], unit='ms')
        if position == "sell" and (data["rsi14"] < 60 and data["k_slow"] < 20 and data["d_slow"] < 20) and data["k_slow"]-5 > data["d_slow"]:
            position = "buy"
            price = data["c"]
            count += 1
            coin_count = invest/data["c"]
            print(position, count, date, data["c"])
            print("coins {:2f}".format(coin_count))

        if position == "buy" and float((highest - source["c"]) / highest) > 0.8:
            # print("{:.2f}".format(float((highest - source["c"]) / highest)))
            # print()
            position = "sell"
            profit = (data["c"] - price) * coin_count
            profits += profit
            if profit <= 0:
                loss += profit
                loss_count += 1

            print(position, count, date, data["c"])
            print("profit {:2f}".format(profit))
            print()

        if position == "buy" and (data["rsi14"] > 60 and data["k_slow"] > 80 and data["d_slow"] > 80) and data["k_slow"] < data["d_slow"]-5:
            position = "sell"
            profit = (data["c"] - price) * coin_count
            profits += profit
            if profit <= 0:
                loss += profit
                loss_count += 1

            print(position, count, date, data["c"])
            print("profit {:2f}".format(profit))
            print()


def main():
    symbol = "TRXUSDT"
    df = pd.read_csv(f"data/current/{symbol}.csv")
    old_columns = ["Time", "Open", "High", "Low", "Close", "Volume"]
    show_columns = ["t", "rsi14", "k_slow", "d_slow"]

    df = df[old_columns]
    df.columns = columns

    # df = df.iloc[-22:]
    for i, row in df.iterrows():
        manipulation(symbol, list(row.values))

    print("invest", invest)
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
