import random
from typing import Optional, Tuple
import numpy as np

from agents.helpers import SavedState, PlayerAction, BoardPiece, NO_PLAYER

def generate_move_alphafour(
        board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, Optional[SavedState]]:
    return 0