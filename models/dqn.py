from torch import nn


# Definition of the netwroks
class DQN(nn.Module):
    # Deep Q Network
    def __init__(self, obs_len, hidden_size, actions_n):
        super(DQN, self).__init__()
        # we might want Conv1d ?
        self.fc_val = nn.Sequential(
            nn.Linear(obs_len, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, actions_n)
        )

    def forward(self, x):
        h = self.fc_val(x)
        return h
