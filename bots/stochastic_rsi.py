import pandas as pd

from bots.base import Base


class StochasticRSI(Base):
    def __init__(self, symbol):
        Base.__init__(self, symbol)

    def action(self, data):
        if self.position == "buy" and data["c"] > self.highest:
            self.highest = data["c"]

        data["date"] = pd.to_datetime(data["t"], unit='ms')
        if self.position == "sell" and (data["rsi14"] < 60 and data["k_slow"] < 30 and data["d_slow"] < 30) and data["k_slow"] - data["d_slow"] > 5:
            self.highest = data["c"]
            self.order(data)
            return 0

        if self.position == "buy" and self.distance_percent(data, self.price) < -0.1:
            self.stop_loss(data, self.price)
            return 0

        if self.position == "buy" and self.distance_percent(data, self.price) > 0.3:
            self.take_profit(data, self.price)
            return 0