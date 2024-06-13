import os.path
# from rl.models.conv_dqn import ConvDQN as Model
from rl.models.conv2d_dqn import ConvDQN as Model
import warnings
from rl.agents.Agent import DQNAgent
from process import load_data
from rl.environments.Environment import DQNEnvironment
from config import settings


def load_conv_dqn_agent(model_path, eps_steps=None):
    model = Model(settings.INPUT_DIM, settings.ACTION_NUMBER, learning_rate=settings.LEARNING_RATE)

    if os.path.exists(model_path):
        print("Loading model...")
        model.load_model(model_path)

    dqn_agent = DQNAgent(
        model,
        model,
        settings.REPLAY_MEM_SIZE,
        settings.BATCH_SIZE,
        settings.GAMMA,
        settings.EPS_START,
        settings.EPS_END,
        eps_steps if eps_steps is not None else settings.EPS_STEPS,
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
