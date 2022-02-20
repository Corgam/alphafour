import torch
import torch.nn as nn
import torch.nn.functional as F

NUMBER_OF_RES_LAYERS = 11


class ConvBlock(nn.Module):
    """
    Convolutional Block Performs initial convolution and batch normalization.
    Input: Connect 4 board
    Output: Tensor
    """
    def __init__(self):
        """
        Initializes the convolutional block.
        """
        super(ConvBlock, self).__init__()
        self.bn = nn.BatchNorm2d(42)
        self.conv = nn.Conv2d(1, 42, kernel_size=(3, 3), stride=(1, 1), padding=1)

    def forward(self, value: torch.Tensor):
        """
        Forwards the given value through the convolutional block
        :param value: value to forward
        :return: new value
        """
        temp = self.conv(value)
        temp = self.bn(temp)
        return F.relu(temp)


class ResBlock(nn.Module):
    """
    Residual block. Performs 3 convolutions and 3 batch normalizations.
    Input: Tensor computed in the convolutional block.
    Output: Tensor
    """
    def __init__(self, in_channels: int, out_channels: int):
        """
        Initializes the residual block.
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        """
        super().__init__()
        self.in_channels, self.out_channels = in_channels, out_channels
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=1,
            bias=False,
        )
        self.bn3 = nn.BatchNorm2d(out_channels)

    def forward(self, value: torch.Tensor):
        """
        Forwards the given value through the residual block
        :param value: value to forward
        :return: new value
        """
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
    """
    Output block for the neural network. Takes All computed data and reduces the dimensions.
    Returns policy vector (1 by 7) and value head (1 by 1).
    """
    def __init__(self):
        """
        Initializes the output block.
        """
        super(OutBlock, self).__init__()
        self.conv = nn.Conv2d(42, 7, (3, 3), stride=(1, 1))
        self.ln = nn.Linear(140, 7)

        self.conv1 = nn.Conv2d(42, 1, (3, 3), stride=(1, 1))
        self.ln1 = nn.Linear(20, 1)

    def forward(self, value: torch.Tensor):
        """
        Forwards the given value through the output block.
        Calculates the final policy and value estimate.
        :param value: value to forward
        :return: policy head, value head
        """
        policy_head = self.conv(value)
        policy_head = policy_head.view(1, -1)
        policy_head = self.ln(policy_head)
        policy_head = torch.softmax(policy_head, dim=1)

        value_head = self.conv1(value)
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
        """
        Initializes the AlphaNet.
        """
        super(AlphaNet, self).__init__()
        self.convLayer = ConvBlock()
        self.resLayers = [ResBlock(42, 42)] * NUMBER_OF_RES_LAYERS
        self.fullLayer = OutBlock()

    def forward(self, board: torch.Tensor):
        """
        Forwards the given value through the residual block
        :param board: the input board to run NN on
        :return: values (consisting of: policy, value estimate)
        """
        value = self.convLayer(board)
        for res_layer in self.resLayers:
            value = res_layer(value)
        results = self.fullLayer(value)
        return results


class AlphaLossFunction(torch.nn.Module):
    """
    Main loss function. Takes into account the difference between the value estimates.
    """
    def __init__(self):
        """
        Initializes the Loss function.
        """
        super(AlphaLossFunction, self).__init__()

    @staticmethod
    def forward(y_value: float, value: float):
        """
        Calculates the difference (value error) between the provided value estimates.
        :param y_value: value estimated by NN
        :param value: value from the training dataset
        :return: value error
        """
        value_error = (value - y_value) ** 2
        return value_error
