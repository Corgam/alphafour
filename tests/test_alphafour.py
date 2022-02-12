# TODO: Test the conversion of the normal board into the neural board with few examples.
# TODO: Test creation of the neural network
# TODO: Test creation of the different layer types
import torch

from agents.agent_alphafour.NN import AlphaNet
from agents.common import string_to_board
from agents.helpers import PLAYER1


def test_alpha_net():
    pass
    # net = AlphaNet()
    # board = string_to_board("""
    # |==============|
    # |    X         |
    # |    O         |
    # |    X X       |
    # |    O X X     |
    # |  O X O O     |
    # |  O O X X     |
    # |==============|
    # |0 1 2 3 4 5 6 |
    # """)
    # neuralBoard = createNeuralBoard(board, PLAYER1)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # input_size = 3
    # num_classes = 7
    # learning_rate = 0.001
    # batch_size = 64
    # num_epochs = 1
    #
    # print(net(neuralBoard).shape)
