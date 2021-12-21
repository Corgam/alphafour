from agents.agent_alphafour.mcts import Node, select_the_best_move
from agents.common import string_to_board
from agents.helpers import PLAYER1


def test_MCTS():
    # TODO: Fix
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
    rootNode = Node(board, PLAYER1)
    best_move = select_the_best_move(rootNode)
    print(best_move)
