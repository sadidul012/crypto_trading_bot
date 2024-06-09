from random import random, randrange

import pandas as pd
import torch
from torch import nn, optim
import torch.nn.functional as F


# Convolutional DQN
class ConvDQN(nn.Module):
    def __init__(self, seq_len_in, actions_n, kernel_size=8, learning_rate=0.001):
        super(ConvDQN, self).__init__()
        self.action_number = actions_n
        n_filters = 64
        max_pool_kernel = 2
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size)
        self.maxPool = nn.MaxPool1d(max_pool_kernel, stride=1)
        self.LRelu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size // 2)
        self.hidden_dim = n_filters * ((((seq_len_in - kernel_size + 1) - max_pool_kernel + 1) - kernel_size // 2 + 1) - max_pool_kernel + 1)

        self.out_layer = nn.Linear(self.hidden_dim, self.action_number)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Agent is using device:\t" + str(self.device))
        self.to(self.device)

    def forward(self, x):
        c1_out = self.conv1(x)
        max_pool_1 = self.maxPool(self.LRelu(c1_out))
        c2_out = self.conv2(max_pool_1)
        max_pool_2 = self.maxPool(self.LRelu(c2_out))
        max_pool_2 = max_pool_2.view(-1, self.hidden_dim)

        return self.LRelu(self.out_layer(max_pool_2))

    def load_model(self, path):
        self.load_state_dict(torch.load(path))

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def compute_state(self, batch, batch_size, double=False):
        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        #
        # non_final_mask is a column vector telling wich state of the sampled is final
        # non_final_next_states contains all the non-final states sampled
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        nfns = [s for s in batch.next_state if s is not None]
        non_final_next_states = torch.cat(nfns).view(len(nfns), -1)
        non_final_next_states = non_final_next_states.unsqueeze(1)

        state_batch = torch.cat(batch.state).view(batch_size, -1)
        state_batch = state_batch.unsqueeze(1)
        action_batch = torch.cat(batch.action).view(batch_size, -1)
        reward_batch = torch.cat(batch.reward).view(batch_size, -1)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.forward(state_batch).gather(1, action_batch)

        next_state_action = None
        if double:
            _, next_state_action = self.forward(state_batch).max(1, keepdim=True)

        return state_action_values, reward_batch, non_final_next_states, non_final_mask, next_state_action

    def compute_state_value(self, non_final_next_states, non_final_mask, reward_batch, batch_size, gamma, double=False, next_state_action=None):
        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        # detach removes the tensor from the graph -> no gradient computation is
        # required
        if double:
            next_state_values = torch.zeros(batch_size, device=self.device).view(batch_size, -1)
        else:
            next_state_values = torch.zeros(batch_size, device=self.device)

        out = self.forward(non_final_next_states)

        if double:
            next_state_values[non_final_mask] = out.gather(1, next_state_action[non_final_mask])
        else:
            next_state_values[non_final_mask] = out.max(1)[0].detach()
            next_state_values = next_state_values.view(batch_size, -1)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * gamma) + reward_batch

        return expected_state_action_values

    def optimize(self, state_action_values, expected_state_action_values):
        # Compute MSE loss
        # expected_state_action_values.unsqueeze(1)
        loss = F.mse_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def select_action(self, state, training, steps_done, eps_steps, eps_end, eps_start):
        if isinstance(state, pd.DataFrame):
            state = torch.tensor(
                [el for el in state['c']],
                device=self.device,
                dtype=torch.float
            )

        state = state.unsqueeze(0).unsqueeze(1)

        sample = random()
        if training:
            if steps_done > eps_steps:
                eps_threshold = eps_end
            else:
                eps_threshold = eps_start
        else:
            eps_threshold = eps_end


        # [Exploitation] pick the best action according to current Q approx.
        if sample > eps_threshold:
            with torch.no_grad():
                # Return the number of the action with highest non normalized probability
                # TODO: decide if diverge from paper and normalize probabilities with
                # softmax or at least compare the architectures
                return torch.tensor([self.forward(state).argmax()], device=self.device, dtype=torch.long)

        # [Exploration]  pick a random action from the action space
        else:
            return torch.tensor([randrange(self.action_number)], device=self.device, dtype=torch.long)
