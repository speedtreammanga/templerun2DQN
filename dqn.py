import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims, device=None):
        super(DeepQNetwork, self).__init__()

        self.conv1 = nn.Conv3d(input_dims[0], 6, kernel_size=(1,3,3))
        self.conv2 = nn.Conv3d(6, 24, kernel_size=(1, 3, 3))
        # self.conv3 = nn.Conv3d(24, 36, kernel_size=(1, 3, 3))
        self.pool = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        fc_input_dims = self.calculate_conv_output_dims(input_dims)

        self.fc1 = nn.Linear(fc_input_dims, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_actions)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')  # if device is None else device
        self.to(self.device)

        self.loss = nn.MSELoss()  # nn.CrossEntropyLoss()
        self.optimizer = T.optim.SGD(self.parameters(), lr=lr)

    def calculate_conv_output_dims(self, input_dims):
        print("input_dims", input_dims)
        state_shape = T.zeros(1, *input_dims)
        print("dims0", state_shape.shape)
        dims = self.conv1(state_shape)
        print("conv1", dims.shape)
        dims = self.pool(dims)
        print("pool1", dims.shape)
        dims = self.conv2(dims)
        print("conv2", dims.shape)
        dims = self.pool(dims)
        print("pool2", dims.shape)
        # dims = self.conv3(dims)
        # print("conv3", dims.shape)
        # dims = self.pool(dims)
        # print("pool3", dims.shape)
        return int(np.prod(dims.size()))

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x.float())))
        x = self.pool(F.relu(self.conv2(x)))
        # x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size()[0], -1)  # flatten conv3 output for fc1 layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    s = (1, 153, 123, 3)
    dqn = DeepQNetwork(0.001, 10, s)
