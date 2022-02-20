import pickle
from typing import Optional, Tuple
import numpy as np
from from_root import from_root

from agents.agent_alphafour.mcts_with_NN import Connect4State, run_alpha_four
from agents.helpers import SavedState, PlayerAction, BoardPiece


def generate_move_alphafour(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    """
    Generates move based on alpha four model
    """
    root_state = Connect4State(board, player)
    file_path = from_root("chosen_iteration.pkl")
    with open(file_path, "rb") as f:
        iteration = pickle.load(f)
    move, root_node = run_alpha_four(root_state, iteration["number_of_mcts_simulations"], iteration["iteration"])
    return move, saved_state
