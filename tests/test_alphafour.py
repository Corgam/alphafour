# TODO: Test the conversion of the normal board into the neural board with few examples.
# TODO: Test creation of the neural network
# TODO: Test creation of the different layer types
from agents.agent_alphafour.neural import AlphaNet, createNeuralBoard
from agents.common import string_to_board


def testAlphaNet():
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
    neuralBoard = createNeuralBoard(board)
    print(net(neuralBoard).shape)
