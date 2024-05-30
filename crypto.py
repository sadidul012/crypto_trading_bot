import pandas as pd
import tqdm
# from bots.stochastic_rsi import StochasticRSI as TheBot
# from bots.grid import Grid as TheBot
from bots.dqn_bot import DQNBot as TheBot
# from metrics import stochastics, rsi
from process import load_data, columns


assets_list = ['FTMUSDT', 'ETHUSDT', "TRXUSDT", "XRPUSDT"]
coins = dict(zip(assets_list, [None] * len(assets_list)))
metrics = dict(zip(assets_list, [None] * len(assets_list)))


def manipulation(trader, source):
    global coins, metrics
    source = dict(zip(columns, source))
    df = pd.DataFrame([source])
    coins[trader.symbol] = pd.concat([coins[trader.symbol].iloc[-23:], df]) if coins[trader.symbol] is not None else df
    df = coins[trader.symbol]
    # df = rsi(df, "c", 10)
    # df = stochastics(df, 'l', 'h', 'c', 14, 3)
    # metrics[trader.symbol] = pd.concat([metrics[trader.symbol].iloc[-17:], df.iloc[-1:]]) if metrics[trader.symbol] is not None else df
    if df.shape[0] >= 23:
        # data = df.iloc[-1].to_dict()
        # trader.action(data)
        trader.action(df)


def main():
    for symbol in assets_list:
        trader = TheBot(symbol)
        df = load_data(symbol)
        progress = tqdm.tqdm(total=df.shape[0], position=0, leave=True, desc=symbol)
        for i, row in df.iterrows():
            manipulation(trader, list(row.values))
            progress.update()

        trader.summary()
        break


if __name__ == '__main__':
    main()
