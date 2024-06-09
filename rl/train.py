import os.path

from process import load_data
from rl.test import load_conv_dqn_agent
from rl.environments.Environment import DQNEnvironment
from config import settings


def main():
    df = load_data('ETHUSDT', replace_column=False)

    model_path = settings.DATA_PATH + settings.MODEL_FOLDER
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    model_path = settings.DATA_PATH + settings.MODEL_LOCATION
    dqn_agent, model = load_conv_dqn_agent(model_path)
    # print(model)

    train_size = int(settings.TRADING_PERIOD * 0.8)
    profit_train_env = DQNEnvironment(df[:train_size], "profit")

    model, cr_profit_dqn = dqn_agent.train(profit_train_env, num_episodes=settings.NUMBER_EPOCHS)
    model.save_model(model_path)
    profit_train_env.reset()


if __name__ == '__main__':
    main()
