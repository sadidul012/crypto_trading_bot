import pandas as pd
import tqdm

# from bots.stochastic_rsi import StochasticRSI as TheBot
from bots.grid import Grid as TheBot
from metrics import stochastics, rsi


assets_list = ['FTMUSDT', 'ETHUSDT', "TRXUSDT", "XRPUSDT"]
coins = dict(zip(assets_list, [None] * len(assets_list)))
metrics = dict(zip(assets_list, [None] * len(assets_list)))
columns = ["t", "o", "h", "l", "c", "v"]


def manipulation(trader, source):
    global coins, metrics, columns
    source = dict(zip(columns, source))
    df = pd.DataFrame([source])
    coins[trader.symbol] = pd.concat([coins[trader.symbol].iloc[-17:], df]) if coins[trader.symbol] is not None else df
    df = coins[trader.symbol]
    df = rsi(df, "c", 10)
    df = stochastics(df, 'l', 'h', 'c', 14, 3)

    metrics[trader.symbol] = pd.concat([metrics[trader.symbol].iloc[-17:], df.iloc[-1:]]) if metrics[trader.symbol] is not None else df
    if df.shape[0] >= 18:
        data = df.iloc[-1].to_dict()
        trader.action(data)


def main():
    import zipfile
    import glob

    for symbol in assets_list:
        trader = TheBot(symbol)

        list_ = []
        for file in glob.glob(f"/mnt/Cache/crypto/{symbol}/*.zip"):
            zf = zipfile.ZipFile(file)
            text_files = zf.infolist()
            for text_file in text_files:
                df = pd.read_csv(zf.open(text_file.filename), header=None)
                list_.append(df)

        df = pd.concat(list_)
        df.sort_values(by=[0], inplace=True)
        df = df.iloc[:100]
        df = df[[0, 1, 2, 3, 4, 5]]
        old_columns = ["Time", "Open", "High", "Low", "Close", "Volume"]
        df.columns = old_columns

        df = df[old_columns]
        df.columns = columns

        progress = tqdm.tqdm(total=df.shape[0], position=0, leave=True, desc=symbol)
        for i, row in df.iterrows():
            manipulation(trader, list(row.values))
            progress.update()

        trader.summary()
        break


if __name__ == '__main__':
    main()
