from bots.botbase import BotBase
from config import settings
from rl.test import load_conv_dqn_agent


class DQNBot(BotBase):
    def __init__(self, symbol):
        BotBase.__init__(self, symbol)
        model_path = settings.DATA_PATH + settings.MODEL_LOCATION
        dqn_agent, _ = load_conv_dqn_agent(model_path)
        self.agent = dqn_agent
        self.actions = ["", "buy", "sell"]

    def action(self, data):
        action = self.actions[self.agent.select_action(data)]
        # if self.position == "buy" and data["c"] > self.highest:
        #     self.highest = data["c"]
        #
        # data["date"] = pd.to_datetime(data["t"], unit='ms')
        data = data.iloc[-1].to_dict()
        data["date"] = data["d"]
        if self.position == "sell" and action == "buy":
            self.highest = data["c"]
            self.order(data)
            return 0

        if self.position == "buy" and action == "sell":
            self.stop_loss(data, self.price)
            return 0

        if self.position == "buy" and action == "sell":
            self.take_profit(data, self.price)
            return 0

        self.last_price = data["c"]
