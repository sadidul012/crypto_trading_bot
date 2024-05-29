import os.path

import torch

from rl.models.conv_dqn import ConvDQN
from utils import load_data
import warnings
from rl.environments.DQNEnvironment import DQNEnvironment
from rl.agents.DQNAgent import DQNAgent


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
    hidden_dim = 1024
    action_number = 3
    target_update = 10
    trading_period = 4000
    model = ConvDQN(input_dim, action_number)
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
            "executing main.py script instead of train_test.py!")

    train_size = int(trading_period * 0.8)
    profit_train_env = DQNEnvironment(df[:train_size], "profit")
    model_path = path + "rl_models/"
    if not os.path.exists(model_path):
        os.makedirs(model_path, exist_ok=True)

    model, cr_profit_dqn = dqn_agent.train(profit_train_env, num_episodes=40)
    torch.save(model.state_dict(), model_path + "profit_reward_dqn_model")
    profit_train_env.reset()


if __name__ == '__main__':
    main()
