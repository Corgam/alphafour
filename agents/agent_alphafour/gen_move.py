import pickle
from typing import Optional, Tuple
import numpy as np
from from_root import from_root

from agents.agent_alphafour.mcts_with_NN import Connect4State, run_AlphaFour
from agents.helpers import SavedState, PlayerAction, BoardPiece


def generate_move_alphafour(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    root_state = Connect4State(board, player)
    filePath = from_root("chosen_iteration.pkl")
    with open(filePath, "rb") as f:
        iteration = pickle.load(f)
    move, root_node = run_AlphaFour(root_state, 1000, iteration["iteration"])
    return move, saved_state
