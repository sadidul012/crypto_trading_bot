import os.path

from rl.test import load_agent
from utils import load_data
from rl.environments.DQNEnvironment import DQNEnvironment
from config import settings


def main():
    print(settings.DATA_PATH)
    df = load_data(settings.DATA_PATH)

    model_path = settings.DATA_PATH + "rl_models/"
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    dqn_agent, model = load_agent()

    train_size = int(settings.TRADING_PERIOD * 0.8)
    profit_train_env = DQNEnvironment(df[:train_size], "profit")

    model, cr_profit_dqn = dqn_agent.train(profit_train_env, num_episodes=40)
    model.save_model(model_path)
    profit_train_env.reset()


if __name__ == '__main__':
    main()
