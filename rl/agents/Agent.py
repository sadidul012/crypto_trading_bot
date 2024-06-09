import numpy as np
from rl.agents.memory import ReplayMemory, Transition
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class DQNAgent:
    """Definition of the Agent that will interact with the environment.

    Attributes:
        replay_mem_size (:obj:`int`): max capacity of Replay Memory

        batch_size (:obj:`int`): Batch size. Default is 40 as specified in the paper.

        gamma (:obj:`float`): The discount, should be a constant between 0 and 1
            that ensures the sum converges. It also controls the importance of future
            expected reward.

        eps_start(:obj:`float`): initial value for epsilon of the e-greedy action
            selection

        eps_end(:obj:`float`): final value for epsilon of the e-greedy action
            selection

        target_update (:obj:`int`): period of Q target network updates

        double (:obj:`bool`): Type of Q function computation.
    """

    def __init__(
        self,
        policy_net,
        target_net,
        replay_mem_size=10000,
        batch_size=40,
        gamma=0.98,
        eps_start=1,
        eps_end=0.12,
        eps_steps=300,
        target_update=10,
        double=True
    ):

        self.replay_mem_size = replay_mem_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.target_update = target_update
        self.double = double  # to understand if use or do not use a 'Double' model (regularization)
        self.training = True  # to do not pick random actions during testing

        self.policy_net = policy_net
        self.target_net = target_net
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.memory = ReplayMemory(self.replay_mem_size)
        self.steps_done = 0
        self.training_cumulative_reward = []

    def select_action(self, state):
        """ the epsilon-greedy action selection"""

        action = self.policy_net.select_action(state, self.training, self.steps_done, self.eps_steps, self.eps_end, self.eps_start)
        self.steps_done += 1
        return action

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            # it will return without doing nothing if we have not enough data to sample
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        # Transition is the named tuple defined above.
        batch = Transition(*zip(*transitions))

        state_action_values, reward_batch, non_final_next_states, non_final_mask, next_state_action = self.policy_net.compute_state(batch, self.batch_size, self.double)
        expected_state_action_values = self.target_net.compute_state_value(non_final_next_states, non_final_mask, reward_batch, self.batch_size, self.gamma, self.double, next_state_action)

        self.policy_net.optimize(state_action_values, expected_state_action_values)

    def train(self, env, num_episodes=40):
        self.training = True
        writer = SummaryWriter('../data/log', filename_suffix="DQN")
        cumulative_reward = [0 for t in range(num_episodes)]
        progress = tqdm(range(num_episodes), desc="Training")
        for i_episode in progress:
            progress.set_postfix({"cumulative_reward": "{:.2f}".format(np.mean(cumulative_reward[:i_episode]))})

            # Initialize the environment and state
            env.reset()  # reset the env st it is set at the beginning of the time series
            self.steps_done = 0
            state = env.get_state()
            for t in range(len(env.data)):  # while not env.done
                # Select and perform an action
                action = self.select_action(state)
                reward, done, _ = env.step(action)

                cumulative_reward[i_episode] += reward.item()

                # Observe new state: it will be None if env.done = True. It is the next
                # state since env.step() has been called two rows above.
                next_state = env.get_state()

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network): note that
                # it will return without doing nothing if we have not enough data to sample

                self.optimize_model()

                if done:
                    break

            # Update the target network, copying all weights and biases of policy_net
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            writer.add_scalar('EpisodeReward/train', cumulative_reward[i_episode], i_episode)
            writer.add_scalar('EpisodeAvgReward/train', np.mean(cumulative_reward[:i_episode]), i_episode)
        writer.close()

        return self.policy_net, cumulative_reward

    def test(self, env_test):
        self.training = False
        cumulative_reward = [0 for t in range(len(env_test.data))]
        reward_list = [0 for t in range(len(env_test.data))]
        env_test.reset()  # reset the env st it is set at the beginning of the time serie
        state = env_test.get_state()
        for t in tqdm(range(len(env_test.data)), desc="Testing"):  # while not env.done
            # Select and perform an action
            action = self.select_action(state)

            reward, done, _ = env_test.step(action)

            cumulative_reward[t] += reward.item() + cumulative_reward[t - 1 if t - 1 > 0 else 0]
            reward_list[t] = reward

            # Observe new state: it will be None if env.done = True. It is the next
            # state since env.step() has been called two rows above.
            next_state = env_test.get_state()
            # Move to the next state
            state = next_state

            if done:
                break

        return cumulative_reward, reward_list
