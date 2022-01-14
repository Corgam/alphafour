from agents.agent_alphafour.mcts import Connect4State, run_MCTS
from agents.common import string_to_board
from agents.helpers import PLAYER1


def test_MCTS():
    board = string_to_board("""
        |==============|
        |              |
        |              |
        |              |
        |        O     |
        |        O     |
        |  X X   O   X |
        |==============|
        |0 1 2 3 4 5 6 |
        """)
    root_state = Connect4State(board, PLAYER1)
    move = run_MCTS(root_state, 100)
    assert move == 4
