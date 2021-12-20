import torch
from typing import Optional, Tuple
import numpy as np
from agents.agent_alphafour.neural import AlphaNet, createNeuralBoard
from agents.common import string_to_board

from agents.helpers import SavedState, PlayerAction, BoardPiece, NO_PLAYER

def generate_move_alphafour(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
        return 0
