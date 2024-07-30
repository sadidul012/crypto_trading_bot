from matplotlib import pyplot as plt

from bots.botbase import BotBase
from metrics import rsi


class RSIBot(BotBase):
    def __init__(self, symbol):
        BotBase.__init__(self, symbol)
        self.rsi = []
        self.closes = []

    def action(self, data):
        data = rsi(data, "c", 10)
        data = data.iloc[-1].to_dict()
        self.rsi.append(data["rsi14"])
        self.closes.append(data["c"])

        # if self.position == "buy" and data["c"] > self.highest:
        #     self.highest = data["c"]
        #
        # data["date"] = pd.to_datetime(data["t"], unit='ms')
        # if self.position == "sell" and (data["rsi14"] < 60 and data["k_slow"] < 30 and data["d_slow"] < 30) and data["k_slow"] - data["d_slow"] > 5:
        #     self.highest = data["c"]
        #     self.order(data)
        #     return 0
        #
        # if self.position == "buy" and self.distance_percent(data, self.price) < -0.1:
        #     self.stop_loss(data, self.price)
        #     return 0
        #
        # if self.position == "buy" and self.distance_percent(data, self.price) > 0.3:
        #     self.take_profit(data, self.price)
        #     return 0

        self.last_price = data["c"]

    def render(self):
        if 260 < len(self.rsi) < 265:
            fig, ax = plt.subplots(2, 1)
            ax[0].plot(self.closes)
            ax[1].plot(self.rsi)
            plt.savefig("rsi.png")
