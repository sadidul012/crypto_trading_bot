import pandas as pd

from bots.base import Base


class StochasticRSI(Base):
    def __init__(self, symbol):
        Base.__init__(self, symbol)

    def action(self, data):
        if self.position == "buy" and data["c"] > self.highest:
            self.highest = data["c"]

        data["date"] = pd.to_datetime(data["t"], unit='ms')
        if self.position == "sell" and (data["rsi14"] < 60 and data["k_slow"] < 20 and data["d_slow"] < 20) and data["k_slow"] - data["d_slow"] > 5:
            self.highest = data["c"]
            self.order(data)
            return 0

        # if self.position == "buy" and self.distance_percent(data, self.highest) < -0.01:
        #     self.highest = 0
        #     self.stop_loss(data, self.highest)
        #     return 0

        if self.position == "buy" and self.distance_percent(data, self.price) < -0.1:
            self.stop_loss(data, self.price)
            return 0

        if self.position == "buy" and self.distance_percent(data, self.price) > 0.3:
            self.take_profit(data, self.price)
            return 0

        # if self.position == "buy" and (data["rsi14"] > 60 and data["k_slow"] > 80 and data["d_slow"] > 80):
        #     self.take_profit(data, self.price)
        #     return 0

        # if position == "buy" and float((highest - source["c"]) / highest) > 0.8:
        #     # print("{:.2f}".format(float((highest - source["c"]) / highest)))
        #     # print()
        #     position = "sell"
        #     profit = (data["c"] - price) * coin_count
        #     profits += profit
        #     if profit <= 0:
        #         loss += profit
        #         loss_count += 1
        #
        #     print(position, count, date, data["c"])
        #     print("profit {:2f}".format(profit))
        #     print()
