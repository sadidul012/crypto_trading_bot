import numpy as np
from prettytable import PrettyTable as PrettyTable

from config import settings
import random
from rl.environments.Environment import DQNEnvironment
from process import load_data
from rl.train import load_conv_dqn_agent


random.seed(0)


def print_stats(model, c_return, t):
    c_return = np.array(c_return).flatten()
    t.add_row([str(model), "%.2f" % np.mean(c_return), "%.2f" % np.amax(c_return), "%.2f" % np.amin(c_return), "%.2f" % np.std(c_return)])


def test():
    # testing
    df = load_data('ETHUSDT', replace_column=False)
    train_size = int(df.shape[0] * 0.8)
    model_path = settings.DATA_PATH + settings.MODEL_LOCATION
    dqn_agent, _ = load_conv_dqn_agent(model_path)
    profit_dqn_return = []
    print(df.head())

    i = 0
    while i < settings.N_TEST:
        print("Test nr. %s" % str(i + 1))
        profit_test_env = DQNEnvironment(df[train_size:], "profit")

        # ProfitDQN
        cr_profit_dqn_test, _ = dqn_agent.test(profit_test_env)
        profit_dqn_return.append(profit_test_env.cumulative_return)
        profit_test_env.print_history()
        profit_test_env.reset()
        i += 1

    t = PrettyTable(["Trading System", "Avg. Return (%)", "Max Return (%)", "Min Return (%)", "Std. Dev."])
    print()
    print_stats("ProfitDQN", profit_dqn_return, t)
    print(t)


if __name__ == '__main__':
    test()
