from agents.agent_alphafour.mcts_with_NN import Connect4State, run_alpha_four
from agents.agent_alphafour.self_play import mcts_self_play
from agents.common import string_to_board
from agents.helpers import PLAYER1


def test_MCTS():
    board = string_to_board(
        """
        |==============|
        |              |
        |              |
        |              |
        |        O     |
        |        O     |
        |  X X   O   X |
        |==============|
        |0 1 2 3 4 5 6 |
        """
    )
    root_state = Connect4State(board, PLAYER1)
    move, root_node = run_alpha_four(root_state, 100)
    assert move == 4


def test_MCTS_play():
    mcts_self_play()
