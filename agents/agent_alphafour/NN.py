import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.common import string_to_board

NUMBER_OF_RES_LAYERS = 1


class Convblock(nn.Module):
    def __init__(self):
        super(Convblock, self).__init__()
        self.bn = nn.BatchNorm2d(42)
        self.conv = nn.Conv2d(1, 42, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, value):
        value = torch.from_numpy(value)
        value = value.type(torch.FloatTensor)

        temp = self.conv(value)  # Apply Convolution
        temp = self.bn(temp)  # Applies Batch Normalization - normalize the mean and variance of data
        temp = F.relu(temp)  # Replaces are negative numbers with 0 -> max(0,x)
        return temp


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        # self.expansion = 4
        self.in_channels, self.out_channels, self.activation = in_channels, out_channels, activation
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, value):
        # Input value dim (1,42,6,7)
        residual_value = value
        # First Conv.
        value = self.conv1(value)  # dim (1,42,8,9)
        value = self.bn1(value)
        value = F.relu(value)
        # Second Conv.
        value = self.conv2(value)  # dim (1,42,8,9)
        value = self.bn2(value)
        value = F.relu(value)
        # Third Conv.
        value = self.conv3(value)  # dim (1,42,10,11)
        value = self.bn3(value)
        # Residual addition
        value += residual_value
        value = F.relu(value)
        return value


class FullBlock(nn.Module):
    def __init__(self):
        super(FullBlock, self).__init__()
        # Value Head
        self.conv_head = nn.Conv2d(42, 1, (3, 3), stride=(1, 1))
        self.ln_head = nn.Linear(162, 1)
        # Policy Head
        self.conv_policy = nn.Conv2d(42, 7, (3, 3), stride=(1, 1))
        self.ln_policy = nn.Linear(1134, 7)

    def forward(self, value):
        # Value dim (1,42,10,11)
        value_head = self.conv_head(value)
        # value_head = F.relu(value_head)
        value_head = value_head.view(1, 162)
        value_head = self.ln_head(value_head)

        policy_head = self.conv_policy(value)  # Dim (1,7,8,9)
        policy_head = policy_head.view(1, 1134)
        policy_head = self.ln_policy(policy_head)
        policy_head = torch.softmax(policy_head, dim=1)

        return policy_head, value_head


class AlphaNet(torch.nn.Module):
    """
    Main class for the deep convolutional residual neural network for the connect four agent.
    Consists of one convolutional layer, followed by NUMBER_OF_RES_LAYERS residual layers and a fully connected layer
    at the end.
    """

    def __init__(self) -> None:
        super(AlphaNet, self).__init__()
        self.convLayer = Convblock()
        self.resLayers = [ResBlock(42, 42)] * NUMBER_OF_RES_LAYERS
        self.fullLayer = FullBlock()

    # Parameters

    # Methods
    def forward(self, values):
        # values is board with dims (6,7)
        expanded_values1 = np.expand_dims(values, 0)  # Dims (1,6,7)
        expanded_values2 = np.expand_dims(expanded_values1, 1)  # Dims (1,1,6,7)
        values = self.convLayer(expanded_values2)  # Dims (1,42,6,7)
        for res_layer in self.resLayers:
            values = res_layer(values)
        values = self.fullLayer(values)
        return values


class AlphaLossFunction(torch.nn.Module):
    # TODO
    def __init__(self):
        pass

    def forward(self):
        pass


if __name__ == "__main__":
    net = AlphaNet()
    board = string_to_board("""
    |==============|
    |    X         |
    |    O         |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """)
    neuralBoard = board
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_size = 3
    num_classes = 7
    learning_rate = 0.001
    batch_size = 64
    num_epochs = 1

    print(net(neuralBoard))
