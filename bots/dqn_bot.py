from bots.botbase import BotBase
from config import settings
from rl.test import load_conv_dqn_agent


class DQNBot(BotBase):
    def __init__(self, symbol):
        BotBase.__init__(self, symbol)
        model_path = settings.DATA_PATH + settings.MODEL_LOCATION
        dqn_agent, _ = load_conv_dqn_agent(model_path, eps_steps=0)
        self.agent = dqn_agent
        self.actions = ["", "buy", "sell"]

    def action(self, data):
        data_copy = data.copy()
        data_copy.columns = ['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        action, confidence = self.agent.select_action(self.agent.policy_net.pd_to_torch(data_copy))
        # print(confidence)
        action = self.actions[action]

        data = data.iloc[-1].to_dict()
        data["date"] = data["d"]
        self.last_price = data["c"]
        # print(self.position, action, confidence)
        if self.position == "sell" and action == "buy":
            # print(self.position, action)
            self.highest = data["c"]
            self.order(data)
            return 0

        if self.price > 0 and self.loss_percent() < -0.04:
            self.stop_loss(data, self.price, close="stop_loss")
            return 0

        # if self.price > 0 and ((self.last_price - self.price) / self.price) > 0.4:
        #     self.take_profit(data, self.price, close="take_profit_force")
        #     return 0

        if self.position == "buy" and action == "sell":
            self.take_profit(data, self.price)
            return 0

        # if self.position == "buy" and action == "sell":
        #     self.take_profit(data, self.price)
        #     return 0
