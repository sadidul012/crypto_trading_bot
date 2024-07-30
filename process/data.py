import pandas as pd
import zipfile
import glob


columns = ["d", "t", "o", "h", "l", "c", "v"]


def load_data(symbol, replace_column=True, daily=False):
    list_ = []
    for file in glob.glob(f"/mnt/Cache/crypto/{symbol}/*.zip"):
        zf = zipfile.ZipFile(file)
        text_files = zf.infolist()
        for text_file in text_files:
            df = pd.read_csv(zf.open(text_file.filename), header=None)
            list_.append(df)

    df = pd.concat(list_)
    df.sort_values(by=[0], inplace=True)
    df = df[[0, 1, 2, 3, 4, 5]]
    old_columns = ["Time", "Open", "High", "Low", "Close", "Volume"]
    df.columns = old_columns
    df["Date"] = pd.to_datetime(df["Time"], unit="ms")

    df = df[["Date"] + old_columns]
    if replace_column:
        df.columns = columns

    if daily:
        df["date"] = df.d.apply(lambda x: pd.to_datetime(x).strftime('%Y/%m/%d'))
        df = df.groupby("date").agg({
            "d": "first",
            "t": "first",
            "o": "first",
            "h": "max",
            "l": "min",
            "c": "last",
            "v": "sum",
        })[columns]
    return df
