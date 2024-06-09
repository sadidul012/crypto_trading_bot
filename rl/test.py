import os

import numpy as np
from prettytable import PrettyTable as PrettyTable

from config import settings
from rl.models.conv_dqn import ConvDQN as Model
# from rl.models.conv2d_dqn import ConvDQN as Model
import random
import warnings
from rl.environments.Environment import DQNEnvironment
from rl.agents.Agent import DQNAgent
from process import load_data


random.seed(0)


def print_stats(model, c_return, t):
    c_return = np.array(c_return).flatten()
    t.add_row([str(model), "%.2f" % np.mean(c_return), "%.2f" % np.amax(c_return), "%.2f" % np.amin(c_return), "%.2f" % np.std(c_return)])


def load_conv_dqn_agent(model_path):
    model = Model(settings.INPUT_DIM, settings.ACTION_NUMBER, learning_rate=settings.LEARNING_RATE)

    # if os.path.exists(model_path):
    #     print("Loading model...")
    #     model.load_model(model_path)

    dqn_agent = DQNAgent(
        model,
        model,
        settings.REPLAY_MEM_SIZE,
        settings.BATCH_SIZE,
        settings.GAMMA,
        settings.EPS_START,
        settings.EPS_END,
        settings.EPS_STEPS,
        settings.TARGET_UPDATE,
        double=settings.DOUBLE
    )
    if str(model.device) == "cpu":
        warnings.warn(
            "Device is set to CPU. This will lead to a very slow training. Consider to run pretained rl by"
            "executing main.py script instead of train_test.py!"
        )

    return dqn_agent, model


def main():
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
    main()
