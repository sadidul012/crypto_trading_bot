import os.path

from prettytable import PrettyTable as PrettyTable
from utils import load_data, print_stats, plot_multiple_conf_interval
import random
import warnings
from Environment import Environment
from Agent import Agent


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
    TRADING_PERIOD = 4000
    index = random.randrange(len(df) - TRADING_PERIOD - 1)
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
    profit_train_env = Environment(df[index:index + train_size], "profit")
    model_path = path + "rl_models/"
    if not os.path.exists(model_path):
        os.makedirs(path, exist_ok=True)
    cr_profit_dqn = dqn_agent.train(profit_train_env, model_path, num_episodes=40)
    profit_train_env.reset()


if __name__ == '__main__':
    main()
