### CODE BY EMIL BALITZKI ###
import random
from enum import Enum
from typing import Callable, Tuple, Optional

import numpy as np

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 (player to move first) has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 (player to move second) has a piece

BoardPiecePrint = str  # dtype for string representation of BoardPiece.
NO_PLAYER_PRINT = BoardPiecePrint(' ')  # Print representation of empty space on the board.
PLAYER1_PRINT = BoardPiecePrint('X')  # Print representation of the Player 1 piece on the board.
PLAYER2_PRINT = BoardPiecePrint('O')  # Print representation of the Player 2 piece on the board.

PlayerAction = np.int8  # The column to be played


class SavedState:
    """
    Class for storing the state of the board for future use.
    """
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


class GameState(Enum):
    """
    Class for representing the game state of the board.
    """
    IS_WIN = 1
    IS_DRAW = -1
    IS_LOST = -2
    STILL_PLAYING = 0


# Check if the given board contains given player's piece at given location
def check_piece(board: np.ndarray, location: tuple, piece: BoardPiece) -> bool:
    """
    Check if there exists a given piece in a specified location at the board.

    :param board: the board to be checked
    :param location: the location on the board in form of tuple(row, column)
    :param piece: the piece to be checked for
    :return: if_piece_exists
    """
    # Check tuple length
    assert len(location) >= 2
    # Check the piece
    return board[location[0], location[1]] == piece


# Converts the piece print to its number notation
def convert_print2number(symbol: BoardPiecePrint):
    """
    Converts the piece's print symbol to its number notation

    :param symbol: the print symbol of the piece to convert
    :return: number_notation_piece
    """
    if symbol == NO_PLAYER_PRINT:
        return NO_PLAYER
    elif symbol == PLAYER1_PRINT:
        return PLAYER1
    elif symbol == PLAYER2_PRINT:
        return PLAYER2


def convert_number2print(piece: BoardPiece):
    """
    Converts the piece's number notation to its print symbol.

    :param piece: piece to convert
    :return: print_symbol
    """
    if piece == NO_PLAYER:
        return NO_PLAYER_PRINT
    elif piece == PLAYER1:
        return PLAYER1_PRINT
    elif piece == PLAYER2:
        return PLAYER2_PRINT


# Get the opposite player
def get_rival_piece(player: BoardPiece):
    """
    Returns the rival piece of a given piece

    :param player: the original piece
    :return: rival_piece
    """
    return PLAYER2 if player == PLAYER1 else PLAYER1


def calculate_possible_moves(board: np.ndarray) -> list[PlayerAction]:
    """
    Generates the list of all moves, for which the columns are not full.

    :param board: the board from which the list will be generated
    :return: list_of_moves containing values of type PlayerAction
    """
    # Always put the middle column first
    columns_order = [3]
    # Then other columns in the order of how far they are from the middle (randomness for the same distance)
    for distance in [1, 2, 3]:
        temp_order = [3 - distance, 3 + distance]
        random.shuffle(temp_order)
        columns_order.extend(temp_order)
    # Check if the columns are not full
    possible_moves = []
    for column_number in columns_order:
        if board[0][column_number] == NO_PLAYER:
            possible_moves.append(np.int8(column_number))
    return possible_moves
