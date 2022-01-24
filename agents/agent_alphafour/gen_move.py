from typing import Optional, Tuple
import numpy as np

from agents.agent_alphafour.mcts import Connect4State, run_MCTS, Node
from agents.helpers import SavedState, PlayerAction, BoardPiece


def generate_move_alphafour(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    root_state = Connect4State(board, player)
    move, root_node = run_MCTS(root_state, 100)
    return move, saved_state


def calculatePolicy(node: Node):
    """
    Calculates policy
    :param node:
    :return:
    """
    sumVisits = 0
    for child in node.children:
        sumVisits += child.visits
    policy = [child.visits / sumVisits for child in node.children]
    return policy
