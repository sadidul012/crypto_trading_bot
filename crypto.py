import pandas as pd
import tqdm
from process import load_data, columns

# from bots.stochastic_rsi import StochasticRSI as TheBot
# from bots.grid import Grid as TheBot
from bots.dqn_bot import DQNBot as TheBot


assets_list = ['FTMUSDT', 'ETHUSDT', "TRXUSDT", "XRPUSDT"]
coins = dict(zip(assets_list, [None] * len(assets_list)))


def manipulation(trader, source):
    global coins
    source = dict(zip(columns, source))
    df = pd.DataFrame([source])
    coins[trader.symbol] = pd.concat([coins[trader.symbol].iloc[-23:], df]) if coins[trader.symbol] is not None else df
    df = coins[trader.symbol]

    if df.shape[0] >= 23:
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
        trader.save_history()
        break


if __name__ == '__main__':
    main()
