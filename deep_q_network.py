import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

# implement: https://www.youtube.com/watch?v=wc-FxNENg9U
# github: https://github.com/philtabor/Deep-Q-Learning-Paper-To-Code/tree/master/DQN
# other simpler implementation: https://github.com/mswang12/minDQN/blob/main/minDQN.py

#
# class DeepQNetwork(nn.Module):
#     def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
#         super(DeepQNetwork, self).__init__()
#
#         self.input_dims = input_dims
#         self.fc1_dims = fc1_dims
#         self.fc2_dims = fc2_dims
#         self.n_actions = n_actions
#         print("tuple", self.input_dims)
#         self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)  # .to(self.device)
#         self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)  # .to(self.device)
#         self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)  # .to(self.device)
#         self.loss = nn.MSELoss()  # .to(self.device)
#         self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
#         # self.device = 'cpu'
#         self.to(self.device)
#         self.optimizer = optim.Adam(self.parameters(), lr=lr)
#
#     def forward(self, state):
#         print("dqn forward", state)
#         x = F.relu(self.fc1(state))
#         x = F.relu(self.fc2(x))
#         actions = self.fc3(x)
#         return actions


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, fc1_dims, fc2_dims):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Conv2d(input_dims[0], 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.RMSprop(self.parameters(), lr=lr)

        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def calculate_conv_output_dims(self, input_dims):
        state = T.zeros(1, *input_dims)
        dims = self.conv1(state)
        dims = self.conv2(dims)
        dims = self.conv3(dims)
        return int(np.prod(dims.size()))

    def forward(self, state):
        conv1 = F.relu(self.conv1(state))
        conv2 = F.relu(self.conv2(conv1))
        conv3 = F.relu(self.conv3(conv2))
        # conv3 shape is BS x n_filters x H x W
        conv_state = conv3.view(conv3.size()[0], -1)
        # conv_state shape is BS x (n_filters * H * W)
        flat1 = F.relu(self.fc1(conv_state))
        actions = self.fc2(flat1)

        return actions
