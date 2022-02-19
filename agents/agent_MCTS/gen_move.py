from typing import Tuple, Optional

import numpy as np

from agents.agent_MCTS.mcts_only import Connect4StateMCTS, run_basic_mcts
from agents.helpers import PlayerAction, SavedState, BoardPiece


def generate_move_mcts(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    root_state = Connect4StateMCTS(board, player)
    move, root_node = run_basic_mcts(root_state, 100)
    return move, saved_state
