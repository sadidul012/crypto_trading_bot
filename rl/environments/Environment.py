import pandas as pd
import torch


# TODO: modify the reward st. we can choose between sharpe ratio reward or profit
# reward as shown in the paper.
class DQNEnvironment:
    """Definition of the trading environment for the DQN-Agent.

    Attributes:
        data (pandas.DataFrame): Time serie to be considered within the environment.

        t (:obj:`int`): Current time instant we are considering.

        profits (:obj:`float`): profit of the agent at time self.t

        agent_positions(:obj:`list` :obj:`float`): list of the positions
           currently owned by the agent.

        agent_position_value(:obj:`float`): current value of open positions
           (positions in self.agent_positions)

        cumulative_return(:obj:`list` :obj:`float`): econometric measure of profit
            during time

        init_price(:obj:`float`): the price of stocks at the beginning of trading
            period.
    """

    def __init__(self, data, reward):
        """
        Creates the environment. Note: Before using the environment you must call
        the Environment.reset() method.

        Args:
           data (:obj:`pd.DataFrane`): Time serie to be initialize the environment.
           reward (:obj:`str`): Type of reward function to use, either sharpe ratio
              "sr" or profit function "profit"
        """
        self.data = data
        self.reward_f = reward if reward == "sr" else "profit"
        self.reset()
        self.t = 23
        self.done = False
        self.profits = []
        self.agent_positions = []
        self.agent_positions_date = []
        self.agent_open_position_value = 0

        self.cumulative_return = []
        self.init_price = 0
        self.history = []

    def print_history(self):
        df = pd.DataFrame(self.history, columns=['buy_date', 'sell_date', 'buy_price', "sell_price", 'profit', "period"])
        df["PNL (%)"] = df["profit"] / df["buy_price"] * 100
        print(df.to_string(index=False))
        print()
        print("Total profit:", df["profit"].sum())
        total_win = (df["profit"] > 0).sum()
        print("Total trades", df.shape[0])
        print("Total win:", total_win)
        print("Total loss:", df.shape[0] - total_win)
        print("Max win:", df["profit"].max())
        print("Max loss:", df["profit"].min())
        print("STD:", df["profit"].median())

    def reset(self):
        """
        Reset the environment or makes a further step of initialization if called
        on an environment never used before. It must always be called before .step()
        method to avoid errors.
        """
        self.t = 23
        self.done = False
        self.profits = [0 for e in range(len(self.data))]
        self.agent_positions = []
        self.agent_positions_date = []
        self.agent_open_position_value = 0

        self.cumulative_return = [0 for e in range(len(self.data))]
        self.init_price = self.data.iloc[0, :]['Close']
        self.history = []

    def get_state(self):
        """
            Return the current state of the environment. NOTE: if called after
            Environment.step() it will return the next state.
        """

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if not self.done:
            return torch.tensor(
                [el for el in self.data.iloc[self.t - 23:self.t + 1, :]['Close']],
                device=device,
                dtype=torch.float
            )
        else:
            return None

    def step(self, act):
        """
        Perform the action of the Agent on the environment, computes the reward
        and update some datastructures to keep track of some econometric indexes
        during time.

        Args:
           act (:obj:`int`): Action to be performed on the environment.

        Returns:
            reward (:obj:`torch.tensor` :dtype:`torch.float`): the reward of
                performing the action on the current env state.
            self.done (:obj:`bool`): A boolean flag telling if we are in a final
                state
            current_state (:obj:`torch.tensor` :dtype:`torch.float`):
                the state of the environment after the action execution.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        reward = 0
        # GET CURRENT STATE
        state = self.data.iloc[self.t, :]['Close']
        date = self.data.iloc[self.t, :]['Date']

        # EXECUTE THE ACTION (act = 0: stay, 1: buy, 2: sell)
        if act == 0:  # Do Nothing
            pass

        if act == 1:  # Buy
            self.agent_positions.append(self.data.iloc[self.t, :]['Close'])
            self.agent_positions_date.append(date)
            reward += 5

        sell_nothing = False
        if act == 2:  # Sell
            profits = 0
            if len(self.agent_positions) < 1:
                sell_nothing = True
            for position in self.agent_positions:
                profits += (self.data.iloc[self.t, :]['Close'] - position)  # profit = close - my_position for each my_position "p"

            if len(self.agent_positions) > 0:
                self.history.append(
                    [
                        self.agent_positions_date[0],
                        date,
                        self.agent_positions[0],
                        self.data.iloc[self.t, :]['Close'],
                        self.data.iloc[self.t, :]['Close'] - self.agent_positions[0],
                        len(self.agent_positions) + 1
                    ]
                )

            self.profits[self.t] = profits
            self.agent_positions = []
            reward += profits * 10
            # print("profit", profits)

        self.agent_open_position_value = 0
        for position in self.agent_positions:
            self.agent_open_position_value += (self.data.iloc[self.t, :]['Close'] - position)
            # TO CHECK if the calculus is correct according to the definition
            self.cumulative_return[self.t] += (position - self.init_price) / self.init_price

        if sell_nothing and (reward > -1):
            reward = -1

        # UPDATE THE STATE
        self.t += 1

        if self.t == len(self.data) - 1:
            self.done = True

        reward += len(self.agent_positions) * 2

        return (
            torch.tensor([reward], device=device, dtype=torch.float),
            self.done,
            torch.tensor([state], dtype=torch.float)
        )  # reward, done, current_state
