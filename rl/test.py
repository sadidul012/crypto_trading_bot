import os.path

import torch
from prettytable import PrettyTable as PrettyTable

from rl import ConvDQN
from utils import load_data, print_stats, plot_multiple_conf_interval
import random
import warnings
from DQNEnvironment import DQNEnvironment
from DQNAgent import DQNAgent


random.seed(0)


def main():
    path = '../data/'
    df = load_data(path)

    replay_mem_size = 10000
    batch_size = 2024
    gamma = 0.98
    eps_start = 1
    eps_end = 0.12
    eps_steps = 300
    learning_rate = 0.001
    input_dim = 24
    hidden_dim = 120
    action_number = 3
    target_update = 10
    n_test = 10
    trading_period = 4000

    model_path = path + "rl_models/profit_reward_dqn_model_1"
    model = ConvDQN(input_dim, action_number)
    model.load_model(model_path)
    dqn_agent = DQNAgent(
        model,
        model,
        'dqn',
        replay_mem_size,
        batch_size,
        gamma,
        eps_start,
        eps_end,
        eps_steps,
        learning_rate,
        input_dim,
        hidden_dim,
        action_number,
        target_update,
        double=False
    )
    if str(dqn_agent.device) == "cpu":
        warnings.warn(
            "Device is set to CPU. This will lead to a very slow training. Consider to run pretained rl by"
            "executing main.py script instead of train_test.py!"
        )

    train_size = int(trading_period * 0.8)
    profit_dqn_return = []

    i = 0
    while i < n_test:
        print("Test nr. %s" % str(i + 1))
        index = random.randrange(len(df) - trading_period - 1)

        profit_test_env = DQNEnvironment(df[index + train_size:index + trading_period], "profit")

        # ProfitDQN
        cr_profit_dqn_test, _ = dqn_agent.test(profit_test_env)
        profit_dqn_return.append(profit_test_env.cumulative_return)
        profit_test_env.reset()
        i += 1

    t = PrettyTable(["Trading System", "Avg. Return (%)", "Max Return (%)", "Min Return (%)", "Std. Dev."])
    print_stats("ProfitDQN", profit_dqn_return, t)
    print(t)
    # plot_multiple_conf_interval(
    #     ["ProfitDQN", "SharpeDQN", "ProfitDDQN", "SharpeDDQN", "ProfitD-DDQN", "SharpeD-DDQN"],
    #     [profit_dqn_return])


if __name__ == '__main__':
    main()
