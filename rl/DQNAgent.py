import torch
import torch.optim as optim
import torch.nn.functional as F
from utils import ReplayMemory
from utils import Transition
import random
from tqdm import tqdm
import os


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

        learning_rate(:obj:`float`): learning rate of the optimizer
            (Adam)

        input_dim (:obj:`int`): input dimentionality withut considering batch size.

        hidden_dim (:obj:`int`): hidden layer dimentionality (for Linear rl only)

        action_number (:obj:`int`): dimentionality of output layer of the Q network

        target_update (:obj:`int`): period of Q target network updates

        model (:obj:`string`): type of the model.

        double (:obj:`bool`): Type of Q function computation.
    """

    def __init__(
        self,
        policy_net,
        target_net,
        model_name,
        replay_mem_size=10000,
        batch_size=40,
        gamma=0.98,
        eps_start=1,
        eps_end=0.12,
        eps_steps=300,
        learning_rate=0.001,
        input_dim=24,
        hidden_dim=120,
        action_number=3,
        target_update=10,
        double=True
    ):

        self.replay_mem_size = replay_mem_size
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_steps = eps_steps
        self.learning_rate = learning_rate
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_number = action_number
        self.target_update = target_update
        self.model = model_name  # deep q network (dqn) or Dueling deep q network (ddqn)
        self.double = double  # to understand if use or do not use a 'Double' model (regularization)
        self.training = True  # to do not pick random actions during testing
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Agent is using device:\t" + str(self.device))

        self.policy_net = policy_net
        self.target_net = target_net
        self.policy_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.learning_rate)
        self.memory = ReplayMemory(self.replay_mem_size)
        self.steps_done = 0
        self.training_cumulative_reward = []

    def select_action(self, state):
        """ the epsilon-greedy action selection"""
        state = state.unsqueeze(0).unsqueeze(1)
        sample = random.random()
        if self.training:
            if self.steps_done > self.eps_steps:
                eps_threshold = self.eps_end
            else:
                eps_threshold = self.eps_start
        else:
            eps_threshold = self.eps_end

        self.steps_done += 1
        # [Exploitation] pick the best action according to current Q approx.
        if sample > eps_threshold:
            with torch.no_grad():
                # Return the number of the action with highest non normalized probability
                # TODO: decide if diverge from paper and normalize probabilities with
                # softmax or at least compare the architectures
                return torch.tensor([self.policy_net(state).argmax()], device=self.device, dtype=torch.long)

        # [Exploration]  pick a random action from the action space
        else:
            return torch.tensor([random.randrange(self.action_number)], device=self.device, dtype=torch.long)

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

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        #
        # non_final_mask is a column vector telling wich state of the sampled is final
        # non_final_next_states contains all the non-final states sampled
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device,
                                      dtype=torch.bool)
        nfns = [s for s in batch.next_state if s is not None]
        non_final_next_states = torch.cat(nfns).view(len(nfns), -1)
        non_final_next_states = non_final_next_states.unsqueeze(1)

        state_batch = torch.cat(batch.state).view(self.batch_size, -1)
        state_batch = state_batch.unsqueeze(1)
        action_batch = torch.cat(batch.action).view(self.batch_size, -1)
        reward_batch = torch.cat(batch.reward).view(self.batch_size, -1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        # detach removes the tensor from the graph -> no gradient computation is
        # required
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        next_state_values = next_state_values.view(self.batch_size, -1)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch
        # print("expected_state_action_values.shape:\t%s"%str(expected_state_action_values.shape))

        # Compute MSE loss
        loss = F.mse_loss(state_action_values,
                          expected_state_action_values)  # expected_state_action_values.unsqueeze(1)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def optimize_double_dqn_model(self):
        if len(self.memory) < self.batch_size:
            # it will return without doing nothing if we have not enough data to sample
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        # Transition is the named tuple defined above.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        #
        # non_final_mask is a column vector telling wich state of the sampled is final
        # non_final_next_states contains all the non-final states sampled
        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool
        )
        nfns = [s for s in batch.next_state if s is not None]
        non_final_next_states = torch.cat(nfns).view(len(nfns), -1)
        non_final_next_states = non_final_next_states.unsqueeze(1)

        state_batch = torch.cat(batch.state).view(self.batch_size, -1)
        state_batch = state_batch.unsqueeze(1)
        action_batch = torch.cat(batch.action).view(self.batch_size, -1)
        reward_batch = torch.cat(batch.reward).view(self.batch_size, -1)
        # print("state_batch shape: %s\nstate_batch[0]:%s\nactionbatch shape: %s\nreward_batch shape: %s"%(str(state_batch.view(40,-1).shape),str(state_batch.view(40,-1)[0]),str(action_batch.shape),str(reward_batch.shape)))

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # ---------- D-DQN Extra Line---------------
        _, next_state_action = self.policy_net(state_batch).max(1, keepdim=True)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the actions given by policynet.
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        # detach removes the tensor from the graph -> no gradient computation is
        # required
        next_state_values = torch.zeros(self.batch_size, device=self.device).view(self.batch_size, -1)

        out = self.target_net(non_final_next_states)
        next_state_values[non_final_mask] = out.gather(1, next_state_action[non_final_mask])
        # next_state_values = next_state_values.view(self.BATCH_SIZE, -1)
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute MSE loss
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def train(self, env, path, num_episodes=40):
        self.training = True
        cumulative_reward = [0 for t in range(num_episodes)]
        print("Training:")
        for i_episode in tqdm(range(num_episodes)):
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

                if self.double:
                    self.optimize_double_dqn_model()
                else:
                    self.optimize_model()

                if done:
                    break

            # Update the target network, copying all weights and biases of policy_net
            if i_episode % self.target_update == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        # save the model
        if self.double:
            model_name = env.reward_f + '_reward_double_' + self.model + '_model'
            count = 0
            while os.path.exists(path + model_name):  # avoid overrinding rl
                count += 1
                model_name = model_name + "_" + str(count)

        else:
            model_name = env.reward_f + '_reward_' + self.model + '_model'
            count = 0
            while os.path.exists(path + model_name):  # avoid overrinding rl
                count += 1
                model_name = model_name + "_" + str(count)

        torch.save(self.policy_net.state_dict(), path + model_name)

        return cumulative_reward

    def test(self, env_test):
        self.training = False
        cumulative_reward = [0 for t in range(len(env_test.data))]
        reward_list = [0 for t in range(len(env_test.data))]
        env_test.reset()  # reset the env st it is set at the beginning of the time serie
        state = env_test.get_state()
        for t in tqdm(range(len(env_test.data))):  # while not env.done

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
