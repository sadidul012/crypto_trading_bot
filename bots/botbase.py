import pandas as pd


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

    def summary(self):
        df = pd.DataFrame(self.history, columns=self.history_columns)
        df["Duration"] = df["Sell Time"] - df["Buy Time"]
        # print(df[self.show_history_columns].to_string(index=False))
        df.to_csv(f"data/output/{self.symbol}.csv", index=False)

        print("invest", self.invest)
        print("cumulative profit", self.profits)
        print("cumulative loss", self.loss)
        print("loss count", self.loss_count)

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
