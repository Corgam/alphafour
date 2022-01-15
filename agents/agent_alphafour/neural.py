import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from agents.helpers import PLAYER1, PLAYER2, BoardPiece

NOF_RESIDUAL_LAYERS = 19

#
INPUT_SIZE = 3

# Convolutional Layer Constants
CONV_OUTPUTS = 128
CONV_INPUTS = 3
CONV_KERN = (3, 3)
CONV_STRIDE = (1, 1)
CONV_PADDING = (1, 1)


# "Same convolution" is a kernel 3*3 with 1*1 padding and stride.
# It will output the same dimentions as input.

class ConvLayer(torch.nn.Module):
    def __init__(self) -> None:
        super(ConvLayer, self).__init__()
        # self.action_size = 7
        self.conv = nn.Conv2d(CONV_INPUTS, CONV_OUTPUTS, CONV_KERN, CONV_STRIDE, CONV_PADDING)
        # self.bn1 = torch.nn.BarchNorm2d(128)

    def forward(self, values):
        values = F.relu(self.conv(values))
        return values
        # s = s.view(-1, 3, 6, 7)
        # s = F.relu(self.bn1(self.conv1(s)))


class ResLayer(torch.nn.Module):
    def __init__(self) -> None:
        super(ResLayer, self).__init__()
        pass

    def forward(self, values):
        pass


class FullLayer(torch.nn.Module):
    def __init__(self) -> None:
        super(FullLayer, self).__init__()
        self.full = nn.Linear(3 * 6 * 7, 7)

    def forward(self, values):
        return self.full(values)


class AlphaNet(torch.nn.Module):
    """
    Main class for the deep convolutional residual neural network for the connect four agent.
    Consists of one convolutional layer, followed by 19 residual layers and a fully connected layer at the end.
    """

    def __init__(self) -> None:
        super(AlphaNet, self).__init__()
        self.convLayer = ConvLayer()
        self.resLayers = [ResLayer()] * NOF_RESIDUAL_LAYERS
        self.fullLayer = FullLayer()

    # Parameters

    # Methods
    def forward(self, values):
        values = self.convLayer(values)
        for layerID in range(19):
            values = self.resLayers[layerID](values)
        values = self.fullLayer(values)
        return values


def createNeuralBoard(board: np.ndarray, player: BoardPiece = PLAYER1):
    """
    Translates the board array into 3D tensor for NN input.
    First and second dimensions will store 1s where there is a piece of the first and second player respectively.
    The last third dimension, will be filled with value of the next player to move: 1 or 2.
    """
    neuralBoard = np.zeros([3, 6, 7]).astype(int)
    # Move the values from the board
    for row in range(6):
        for col in range(7):
            # First player
            if board[row][col] == PLAYER1:
                neuralBoard[0][row][col] = 1
            # Second player
            elif board[row][col] == PLAYER2:
                neuralBoard[1][row][col] = 1
    # Set the player to move
    neuralBoard[2][:][:] = player
    return neuralBoard
