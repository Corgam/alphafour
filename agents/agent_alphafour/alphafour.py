from typing import Optional, Tuple
import numpy as np

from agents.agent_alphafour.mcts import Node, select_the_best_move

from agents.helpers import SavedState, PlayerAction, BoardPiece


def generate_move_alphafour(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    rootNode = Node(board, player)
    return select_the_best_move(rootNode), saved_state
