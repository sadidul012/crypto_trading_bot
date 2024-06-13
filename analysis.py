import plotly.graph_objects as go

import pandas as pd
from datetime import datetime

from config import settings
from process import load_data


def main():
    pair = 'ETHUSDT'
    df = load_data(pair, replace_column=False)
    history = pd.read_csv(settings.DATA_PATH + "output/" + pair + ".csv")

    fig = go.Figure(data=[go.Candlestick(x=df['Date'], open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'])])

    for i, value in history.iterrows():
        fig.add_annotation(x=value["Buy Time"], y=df.loc[df["Date"] == value["Buy Time"], "Close"], text="B", showarrow=True, arrowhead=1)
        fig.add_annotation(x=value["Sell Time"], y=df.loc[df["Date"] == value["Sell Time"], "Close"], text="S {:.1f}".format(value["PNL"]), showarrow=True, arrowhead=1)

    fig.show()


if __name__ == '__main__':
    main()
