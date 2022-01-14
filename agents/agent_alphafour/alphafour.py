from typing import Optional, Tuple
import numpy as np

from agents.agent_alphafour.mcts import Connect4State, run_MCTS
from agents.helpers import SavedState, PlayerAction, BoardPiece


def generate_move_alphafour(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    root_state = Connect4State(board, player)
    move = run_MCTS(root_state, 100)
    return move, saved_state
