from torch import nn


# Convolutional DQN
class ConvDQN(nn.Module):
    def __init__(self, seq_len_in, actions_n, kernel_size=8):
        super(ConvDQN, self).__init__()
        n_filters = 64
        max_pool_kernel = 2
        self.conv1 = nn.Conv1d(1, n_filters, kernel_size)
        self.maxPool = nn.MaxPool1d(max_pool_kernel, stride=1)
        self.LRelu = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(n_filters, n_filters, kernel_size // 2)

        self.hidden_dim = n_filters * ((((
                                                     seq_len_in - kernel_size + 1) - max_pool_kernel + 1) - kernel_size // 2 + 1) - max_pool_kernel + 1)

        self.out_layer = nn.Linear(self.hidden_dim, actions_n)

    def forward(self, x):
        c1_out = self.conv1(x)
        max_pool_1 = self.maxPool(self.LRelu(c1_out))
        c2_out = self.conv2(max_pool_1)
        max_pool_2 = self.maxPool(self.LRelu(c2_out))
        #    print("c1_out:\t%s"%str(c1_out.shape))
        #    print("max_pool_1:\t%s"%str(max_pool_1.shape))
        #    print("c2_out:\t%s"%str(c2_out.shape))
        #    print("max_pool_2:\t%s"%str(max_pool_2.shape))

        max_pool_2 = max_pool_2.view(-1, self.hidden_dim)
        #    print("max_pool_2_view:\t%s"%str(max_pool_2.shape))

        return self.LRelu(self.out_layer(max_pool_2))
