from bots.botbase import BotBase


class Grid(BotBase):
    def __init__(self, symbol):
        super().__init__(symbol=symbol)

        self.grid_numbers = 10
        self.invest_per_grid = self.invest / self.grid_numbers
        self.grid_distance = 0.01
        self.current_price = 0.4132
        self.grid_distance_amount = self.grid_distance * self.current_price
        self.buy_grids = [self.current_price - (x * self.grid_distance_amount) for x in range(1, 6)]
        self.sell_grids = [self.current_price + (x * self.grid_distance_amount) for x in range(1, 6)]
        self.available_crypto = 0
        self.available_investment = self.invest
        self.stop_loss_value = 0.39
        print(self.buy_grids)
        print(self.sell_grids)

    def summary(self):
        print("total invest", self.invest)
        print("available crypto", self.available_crypto)
        print("available crypto in USDT", self.available_crypto * self.last_price)
        print("available invest", self.available_investment)
        print("total assets", (self.available_crypto * self.last_price) + self.available_investment)
        print("time spent", self.time)

    def buy_order_amount(self, data, invest_amount):
        if self.available_investment >= invest_amount:
            self.available_crypto += invest_amount / data["c"]
            self.available_investment -= invest_amount

    def sell_order_amount(self, data, invest_amount):
        if self.available_crypto * data["c"] > invest_amount:
            self.available_crypto -= invest_amount / data["c"]
            self.available_investment += invest_amount

    # def stop_loss(self):
    #     self.available_crypto

    def action(self, data):
        data = data.iloc[-1].to_dict()

        self.time += 1
        if self.time == 1:
            self.buy_order_amount(data, self.invest_per_grid * len(self.sell_grids))

        if len(self.buy_grids) > 0 and data["c"] < self.buy_grids[0]:
            grid_price = self.buy_grids.pop(0)
            self.buy_order_amount(data, self.invest_per_grid)
            self.sell_grids = [grid_price] + self.sell_grids

        if len(self.sell_grids) > 0 and data["c"] > self.sell_grids[0]:
            grid_price = self.sell_grids.pop(0)
            self.sell_order_amount(data, self.invest_per_grid)
            self.buy_grids = [grid_price] + self.buy_grids

        self.last_price = data["c"]
        # if data["c"] > self.stop_loss_value:
        #     self.stop_loss()
