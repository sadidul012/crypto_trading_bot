import os.path

from prettytable import PrettyTable as PrettyTable
from utils import load_data, print_stats, plot_multiple_conf_interval
import random
import warnings
from Environment import Environment
from Agent import Agent


random.seed(0)


def main():
    path = '../data/'
    df = load_data(path)

    REPLAY_MEM_SIZE = 10000
    BATCH_SIZE = 2024
    GAMMA = 0.98
    EPS_START = 1
    EPS_END = 0.12
    EPS_STEPS = 300
    LEARNING_RATE = 0.001
    INPUT_DIM = 24
    HIDDEN_DIM = 120
    ACTION_NUMBER = 3
    TARGET_UPDATE = 10
    N_TEST = 10
    TRADING_PERIOD = 4000
    dqn_agent = Agent(
        REPLAY_MEM_SIZE,
        BATCH_SIZE,
        GAMMA,
        EPS_START,
        EPS_END,
        EPS_STEPS,
        LEARNING_RATE,
        INPUT_DIM,
        HIDDEN_DIM,
        ACTION_NUMBER,
        TARGET_UPDATE,
        MODEL='dqn',
        DOUBLE=False
    )
    if str(dqn_agent.device) == "cpu":
        warnings.warn(
            "Device is set to CPU. This will lead to a very slow training. Consider to run pretained rl by"
            "executing main.py script instead of train_test.py!")

    train_size = int(TRADING_PERIOD * 0.8)
    profit_dqn_return = []
    model_path = path + "rl_models/"

    i = 0
    while i < N_TEST:
        print("Test nr. %s" % str(i + 1))
        index = random.randrange(len(df) - TRADING_PERIOD - 1)

        profit_test_env = Environment(df[index + train_size:index + TRADING_PERIOD], "profit")

        # ProfitDQN
        cr_profit_dqn_test, _ = dqn_agent.test(profit_test_env, model_name="profit_reward_dqn_model", path=model_path)
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
