### CODE BY EMIL BALITZKI ###
import functools
from typing import Optional
import numpy as np
from numba import njit

from agents.helpers import convert_print2number, convert_number2print
from agents.helpers import NO_PLAYER, PLAYER1, PLAYER2
from agents.helpers import BoardPiece, PlayerAction, GameState


def initialize_game_state() -> np.ndarray:
    """
    Returns an ndarray, shape (6, 7) and data type (dtype) BoardPiece, initialized to 0 (NO_PLAYER).

    :return: new_board
    """
    # Create the ndarray with correct size and data type
    board = np.ndarray((6, 7), BoardPiece)
    # Fill the array with NO_PLAYER values
    board.fill(NO_PLAYER)
    return board


def pretty_print_board(board: np.ndarray) -> str:
    """
    Should return `board` converted to a human readable string representation,
    to be used when playing or printing diagnostics to the console (stdout). The piece in
    board[0, 0] should appear in the lower-left. Here's an example output, note that we use
    PLAYER1_Print to represent PLAYER1 and PLAYER2_Print to represent PLAYER2):
    |==============|
    |              |
    |              |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |

    :param board: the board in the array form
    :return: string_board
    """
    finalstring = "\n|==============|\n"
    # Print every row
    for row in range(6):
        finalstring = finalstring + "|"
        for column in range(7):
            finalstring = finalstring + convert_number2print(board[row][column]) + " "
        finalstring = finalstring + "|\n"
    finalstring = finalstring + "|==============|\n|0 1 2 3 4 5 6 |\n"
    return finalstring


def string_to_board(pp_board: str) -> np.ndarray:
    """
    Takes the output of pretty_print_board and turns it back into an ndarray.
    This is quite useful for debugging, when the agent crashed and you have the last
    board state as a string.

    :param pp_board: the board in the string format
    :return: board_as_array
    """
    # Initialize board
    board = initialize_game_state()
    # Cut off the first board border
    pp_board = pp_board.split("\n", 2)[2]
    # For every line add pieces to the board
    row = 0
    # For every row
    for i in range(6):
        new_board = pp_board.split('\n', 1)
        pp_board = new_board[1]
        line = new_board[0]
        # Fill in the board with the numbers
        line = line.split('|', 2)[1]
        every_second_char, column = False, 0
        # For every char fill in the board
        for char in line:
            # Every second char is a decorative space, skip it.
            if every_second_char:
                every_second_char = False
                continue
            # Fill in the board
            board[row][column] = convert_print2number(char)
            column = column + 1
            # Check if read all symbols in a row
            if column == 7:
                break
            every_second_char = True
        # Go to the next row
        row = row + 1
    return board


@njit()  # Decorator for making the compiled code faster
def apply_player_action(
        board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.
    """
    if copy:
        board = board.copy()

    coord = get_coords(board, action)
    if not copy:
        global lastmove
        lastmove = action

    board[coord[0], action] = player
    return board


def get_coords(board: np.ndarray, piece: PlayerAction):
    row = 0
    while board[row, piece] != NO_PLAYER:

        if row == 5:
            break
        row += 1

    return [row, piece]


@njit()  # Decorator for making the compiled code faster
def connected_four(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.
    """

    coords = get_coords(board, lastmove)
    row = coords[0]
    col = coords[1]
    # The lambda reduces all diagonals returned by diagonal to a single truth value mapped over contains4
    return contains4(board[row - 1], player) or contains4(board[:, col], player) or \
           functools.reduce(lambda a, b: a or contains4(b, player), diagonal(board, lastmove, player), False)

def diagonal(board: np.ndarray, piece: PlayerAction, player: BoardPiece):
    coords = get_coords(board, piece)
    col = piece
    row = coords[0] - 1

    flippedRow = 6 - row

    # if col bigger than row then above main diag in this case do col - row. reverse in other case.
    offset = col - row
    offset2 = flippedRow - col

    diag1 = board.diagonal(offset)
    diag2 = np.fliplr(board).diagonal(offset2)

    return [diag1, diag2]

def contains4(boardSlice, player):
    count = 0

    for entry in boardSlice:

        if entry == player:
            count += 1
        else:
            # found a break in the line... resetting count
            count = 0
        if count == 4:
            return True

    return False

def check_end_state(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING)?
    """
    try:
        if connected_four(board, player, lastmove):
            return GameState.IS_WIN
        if np.count_nonzero(board == 0) != 0:
            return GameState.STILL_PLAYING
        else:
            return GameState.IS_DRAW
    except NameError:
        return GameState.STILL_PLAYING


def if_game_ended(board: np.ndarray, last_action: Optional[PlayerAction] = None) -> bool:
    """
    Used to check if a game with given board has ended: either by winning, loosing or draw.

    :param board: the board to be checked
    :param last_action: optional last action (column) to optimize the search
    :return if_ended
    """
    return check_end_state(board, PLAYER1, last_action) != GameState.STILL_PLAYING
