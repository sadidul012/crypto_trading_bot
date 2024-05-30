import os

from prettytable import PrettyTable as PrettyTable

from config import settings
from rl.models.conv_dqn import ConvDQN
from utils import load_data, print_stats
import random
import warnings
from rl.environments.DQNEnvironment import DQNEnvironment
from rl.agents.DQNAgent import DQNAgent


random.seed(0)


def load_agent():
    model_path = settings.DATA_PATH + "rl_models/profit_reward_dqn_model"
    model = ConvDQN(settings.INPUT_DIM, settings.ACTION_NUMBER)

    if os.path.exists(model_path):
        print("Loading model...")
        model.load_model(model_path)

    dqn_agent = DQNAgent(
        model,
        model,
        settings.MODEL_NAME,
        settings.REPLAY_MEM_SIZE,
        settings.BATCH_SIZE,
        settings.GAMMA,
        settings.EPS_START,
        settings.EPS_END,
        settings.EPS_STEPS,
        settings.LEARNING_RATE,
        settings.INPUT_DIM,
        settings.HIDDEN_DIM,
        settings.ACTION_NUMBER,
        settings.TARGET_UPDATE,
        double=settings.DOUBLE
    )
    if str(dqn_agent.device) == "cpu":
        warnings.warn(
            "Device is set to CPU. This will lead to a very slow training. Consider to run pretained rl by"
            "executing main.py script instead of train_test.py!"
        )

    return dqn_agent, model


def main():
    df = load_data(settings.DATA_PATH)
    train_size = int(settings.TRADING_PERIOD * 0.8)
    dqn_agent, _ = load_agent()
    profit_dqn_return = []

    i = 0
    while i < settings.N_TEST:
        print("Test nr. %s" % str(i + 1))
        profit_test_env = DQNEnvironment(df[train_size:settings.TRADING_PERIOD], "profit")

        # ProfitDQN
        cr_profit_dqn_test, _ = dqn_agent.test(profit_test_env)
        profit_dqn_return.append(profit_test_env.cumulative_return)
        profit_test_env.print_history()
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
