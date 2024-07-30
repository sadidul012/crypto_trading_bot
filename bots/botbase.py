import os.path

import pandas as pd
# import mplfinance as mpf


class BotBase(object):
    def __init__(self, symbol):
        self.symbol = symbol
        self.position = "sell"
        self.count = 0
        self.price = 0
        self.coin_count = 0
        self.profits = 0
        self.loss = 0
        self.loss_count = 0
        self.invest = 100
        self.highest = 0
        self.time = 0
        self.last_price = 0

        self.history = []
        self.buy = None
        self.history_columns = [
            "Buy Time", "Order number", "Buy Price", "Coins", "Sell Time", "Sell Price", "PNL",
            "Change", "Value %", "Close Type"
        ]
        self.show_history_columns = [
            "Buy Time", "Sell Time", "Order number", "Coins", "Buy Price", "Sell Price", "PNL",
            "Change", "Value %", "Close Type", "Duration"
        ]

    def save_history(self):
        df = pd.DataFrame(self.history, columns=self.history_columns)
        df["Duration"] = df["Sell Time"] - df["Buy Time"]
        # print(df[self.show_history_columns].to_string(index=False))
        if not os.path.exists("data/output"):
            os.makedirs("data/output", exist_ok=True)

        df.to_csv(f"data/output/{self.symbol}.csv", index=False)
        print(df[["Buy Price", "Sell Price", "PNL", "Change", "Value %", "Duration"]].describe().to_string())

    def current_asset_value(self):
        return self.coin_count * self.last_price

    def loss_percent(self):
        return (self.current_asset_value() - self.invest) / self.invest

    def summary(self):
        print("invest", self.invest)
        print("cumulative profit", self.profits)
        print("available crypto", self.coin_count)
        print("available crypto in USDT", self.current_asset_value())
        print("total assets", (self.coin_count * self.last_price) + self.profits)
        print("cumulative loss", self.loss)
        print("loss count", self.loss_count)
        print("trade count", self.count)

    def stop_loss(self, data, base_price, close="stop"):
        self.position = "sell"
        profit = (data["c"] - self.price) * self.coin_count
        self.profits += profit
        if profit <= 0:
            self.loss += profit
            self.loss_count += 1
        self.history.append(
            self.buy + [
                data["date"],
                data["c"],
                profit,
                data["c"] - self.price,
                self.distance_percent(data, base_price) * 100,
                close
            ]
        )

    def take_profit(self, data, base_price, close="take profit"):
        self.stop_loss(data, base_price, close)

    def order(self, data):
        self.position = "buy"
        self.price = data["c"]
        self.count += 1
        self.coin_count = self.invest / data["c"]
        self.buy = [data["date"], self.count, self.price, self.coin_count]

    @staticmethod
    def distance_percent(data, base_price):
        return float(data["c"] - base_price) / data["c"]

    def action(self, data):
        return 0

    def render(self):
        pass
    # def render(self):
    #     mpf.plot(self.df, type='candle', style='yahoo', volume=True)
