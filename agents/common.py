### CODE BY EMIL BALITZKI ###
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
        board: np.ndarray, column: PlayerAction, player: BoardPiece, copy: bool = False
) -> np.ndarray:
    """
    Sets board[i, action] = player, where i is the lowest open row. The modified
    board is returned. If copy is True, makes a copy of the board before modifying it.

    :param board: the board to be modified
    :param column: the column where the piece will be placed
    :param player: the player piece which will be placed
    :param copy: if the move should be done on the new copy of the board
    :return: modified_board
    """
    # Check if to copy
    if copy:
        board = board.copy()
    # Check for exceptions
    assert player == 1 or player == 2
    assert 0 <= column <= 6
    # Calculate how many pieces are already in the column
    row = 0
    for i in range(6):
        if board[i][column] != NO_PLAYER:
            row = row + 1
    # If the column is already full, return unmodified board
    if row == 6:
        return board
    # Add the new piece
    board[5 - row][column] = player
    return board


@njit()  # Decorator for making the compiled code faster
def connected_four(board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None) -> bool:
    """
    Returns True if there are four adjacent pieces equal to `player` arranged
    in either a horizontal, vertical, or diagonal line. Returns False otherwise.
    If desired, the last action taken (i.e. last column played) can be provided
    for potential speed optimisation.

    :param board: the board to be checked
    :param player: the player piece for which it will check
    :param last_action: optional last action (column) to optimize the search
    :return: if_connected_four
    """
    # The ranges are defined in a form of a list and not range() function, due to the limitation od the numba njit.
    horizontal_columns_range = [0, 1, 2, 3, 4, 5, 6]
    vertical_columns_range = [0, 1, 2, 3, 4, 5, 6]
    diagonal_left_columns_range = [3, 4, 5, 6]
    diagonal_right_columns_range = [0, 1, 2, 3]
    # Check if the last action option is not none and optimize the search
    if last_action is not None:
        vertical_columns_range = [np.int64(last_action)]  # Cut the range just to one column
        # Filter out the columns which are further then 3 columns from the last action column
        horizontal_columns_range = [col for col in horizontal_columns_range if abs(col - np.int64(last_action)) <= 3]
        # Filter out the columns from original range which are further then 3 columns from the last action column
        diagonal_left_columns_range = \
            [col for col in diagonal_left_columns_range if abs(col - np.int64(last_action)) <= 3]
        # Filter out the columns from original range which are further then 3 columns from the last action column
        diagonal_right_columns_range = \
            [col for col in diagonal_right_columns_range if abs(col - np.int64(last_action)) <= 3]
    # Horizontal win
    for row in [0, 1, 2, 3, 4, 5]:
        pieces = 0
        for column in horizontal_columns_range:
            if board[row][column] == player:
                pieces = pieces + 1
                if pieces == 4:
                    return True
            else:
                pieces = 0
    # Vertical win
    for column in vertical_columns_range:
        pieces = 0
        for row in [0, 1, 2, 3, 4, 5]:
            if board[row][column] == player:
                pieces = pieces + 1
                if pieces == 4:
                    return True
            else:
                pieces = 0
    # Diagonal left
    for row in [3, 4, 5]:
        for column in diagonal_left_columns_range:
            # Check if the first field is one from the looked player
            if board[row][column] == player:
                # Check for another 3 pieces
                pieces = 1
                for i in [1, 2, 3]:
                    if board[row - i][column - i] == player:
                        pieces = pieces + 1
                    else:
                        pieces = 1
                    # Check if found 4 pieces
                    if pieces == 4:
                        return True
    # Diagonal right
    for row in [3, 4, 5]:
        for column in diagonal_right_columns_range:
            # Check if the first field is one from the looked player
            if board[row][column] == player:
                # Check for another 3 pieces
                pieces = 1
                for i in [1, 2, 3]:
                    if board[row - i][column + i] == player:
                        pieces = pieces + 1
                    else:
                        pieces = 1
                    # Check if found 4 pieces
                    if pieces == 4:
                        return True
    return False


def check_end_state(
        board, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    Returns the current game state for the current `player`, i.e. has their last
    action won (GameState.IS_WIN) or drawn (GameState.IS_DRAW) the game,
    or is play still on-going (GameState.STILL_PLAYING).

    :param board: the board to be checked
    :param player: the player piece for which it will check
    :param last_action: optional last action (column) to optimize the search
    :return: game_state
    """
    # If given player has connected four, game is won.
    if connected_four(board, player, last_action):
        return GameState.IS_WIN
    # If no player won the game and there is at least one free space, game is still playing
    elif (not connected_four(board, PLAYER1, last_action)) and (not connected_four(board, PLAYER2, last_action)) \
            and (np.any(board == NO_PLAYER)):
        return GameState.STILL_PLAYING
    # If nobody won and there are no spaces left, game is draw.
    elif np.all(board != NO_PLAYER):
        return GameState.IS_DRAW
    # If rival won
    else:
        return GameState.IS_LOST


def if_game_ended(board: np.ndarray, last_action: Optional[PlayerAction] = None) -> bool:
    """
    Used to check if a game with given board has ended: either by winning, loosing or draw.

    :param board: the board to be checked
    :param last_action: optional last action (column) to optimize the search
    :return if_ended
    """
    return check_end_state(board, PLAYER1, last_action) != GameState.STILL_PLAYING
