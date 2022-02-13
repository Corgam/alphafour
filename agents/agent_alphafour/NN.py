import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.common import string_to_board

NUMBER_OF_RES_LAYERS = 11


class Convblock(nn.Module):
    def __init__(self):
        super(Convblock, self).__init__()

        self.bn = nn.BatchNorm2d(42)
        self.conv = nn.Conv2d(1, 42, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, value):
        # value = torch.from_numpy(value)
        # value = value.type(torch.FloatTensor)
        temp = self.conv(value)
        temp = self.bn(temp)
        return F.relu(temp)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, value):
        # Remember the value
        residual_value = value
        # First conv
        value = self.conv1(value)
        value = self.bn1(value)
        value = F.relu(value)
        # Second conv
        value = self.conv2(value)
        value = self.bn2(value)
        value = F.relu(value)
        # Third conv
        value = self.conv3(value)
        value = self.bn3(value)
        # Add the residual value
        value += residual_value
        value = F.relu(value)
        return value


class OutBlock(nn.Module):
    def __init__(self):
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(42, 7, (3, 3), stride=(1, 1))
        # self.pool = nn.AvgPool2d(kernel_size=(3, 3))
        self.ln = nn.Linear(140, 7)

        self.conv1 = nn.Conv2d(42, 1, (3, 3), stride=(1, 1))
        self.ln1 = nn.Linear(20, 1)

    def forward(self, value):
        policy_head = self.conv(value)
        # policy_head = F.relu(policy_head)
        policy_head = policy_head.view(1, -1)
        policy_head = self.ln(policy_head)
        policy_head = torch.softmax(policy_head, dim=1)

        value_head = self.conv1(value)
        # value_head = F.relu(value_head)
        value_head = value_head.view(1, -1)
        value_head = self.ln1(value_head)

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
        self.fullLayer = OutBlock()

    # Parameters

    # Methods
    def forward(self, values):
        values = self.convLayer(values)
        for res_layer in self.resLayers:
            values = res_layer(values)
        values = self.fullLayer(values)
        return values


class AlphaLossFunction(torch.nn.Module):
    def __init__(self):
        super(AlphaLossFunction, self).__init__()

    def forward(self, y_value, value):
        value_error = (value - y_value) ** 2
        return value_error


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
