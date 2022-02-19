### CODE BY EMIL BALITZKI ###
from typing import Optional

import numpy as np
from agents.helpers import BoardPiece, GameState, check_piece
from agents.helpers import NO_PLAYER, PLAYER1, PLAYER2
from agents.common import (
    initialize_game_state,
    check_end_state,
    apply_player_action,
    string_to_board,
    pretty_print_board,
    connected_four,
    if_game_ended,
)


## initialize_game_state tests ##


def test_initialize_game_state():
    board = initialize_game_state()

    assert isinstance(board, np.ndarray)
    assert board.dtype == BoardPiece
    assert board.shape == (6, 7)
    assert np.all(board == NO_PLAYER)


## pretty_print_board tests ##


def test_pretty_board_from_string():
    # Create a board from data
    data = [
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 2, 1, 1, 0, 0],
        [0, 2, 1, 2, 2, 0, 0],
        [0, 2, 2, 1, 1, 0, 0],
    ]
    board_from_data = np.array(data, BoardPiece)
    # Create the same board from string
    board_from_string = string_to_board(
        """
    |==============|
    |    X         |
    |    O         |
    |    X X       |
    |    O X X     |
    |  O X O O     |
    |  O O X X     |
    |==============|
    |0 1 2 3 4 5 6 |
    """
    )
    assert np.array_equal(board_from_string, board_from_data)


## string_to_board tests ##


def test_pretty_print():
    # Create a board from data
    data = [
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 2, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [0, 0, 2, 1, 1, 0, 0],
        [0, 2, 1, 2, 2, 0, 0],
        [0, 2, 2, 1, 1, 0, 0],
    ]
    board_from_data = np.array(data, BoardPiece)
    # Print the board
    string_from_function = pretty_print_board(board_from_data)
    string_from_text = (
        "\n|==============|\n|    X         |\n|    O         |\n|    X X       |\n|"
        "    O X X     |\n|  O X O O     |\n|  O O X X     |\n|==============|\n|0 1 2 3 4 5 6 |\n"
    )
    assert string_from_function == string_from_text


## apply_player_action tests ##


def test_player_action_simple():
    board = initialize_game_state()
    apply_player_action(board, np.int8(2), PLAYER1, False)
    assert check_piece(board, (5, 2), PLAYER1)
    apply_player_action(board, np.int8(2), PLAYER2, False)
    assert check_piece(board, (4, 2), PLAYER2)
    apply_player_action(board, np.int8(3), PLAYER1, False)
    assert check_piece(board, (5, 3), PLAYER1)
    apply_player_action(board, np.int8(4), PLAYER2, False)
    assert check_piece(board, (5, 4), PLAYER2)
    apply_player_action(board, np.int8(2), PLAYER1, False)
    assert check_piece(board, (3, 2), PLAYER1)
    apply_player_action(board, np.int8(3), PLAYER2, False)
    assert check_piece(board, (4, 3), PLAYER2)


# Check if the copy option works and does not modify the old board
def test_player_action_copy_option():
    board = initialize_game_state()
    new_board = apply_player_action(board, np.int8(2), PLAYER1, True)
    assert np.all(board == NO_PLAYER)
    assert check_piece(new_board, (5, 2), PLAYER1)
    assert not np.array_equal(board, new_board)


def test_player_action_on_full_column():
    board = string_to_board(
        """
            |==============|
            |  X           |
            |  O           |
            |  X           |
            |  O           |
            |  X X X       |
            |  X X O O O   |
            |==============|
            |0 1 2 3 4 5 6 |
            """
    )
    copy_board = board.copy()
    apply_player_action(board, np.int8(1), PLAYER2, False)
    assert np.array_equal(board, copy_board)


## connected_four tests ##


def test_connected4_horizontal():
    board = string_to_board(
        """
            |==============|
            |              |
            |              |
            |              |
            |              |
            |  X X X       |
            |  X X O O O   |
            |==============|
            |0 1 2 3 4 5 6 |
            """
    )
    assert not (connected_four(board, PLAYER1))
    apply_player_action(board, np.int8(4), PLAYER1)
    assert connected_four(board, PLAYER1)
    assert connected_four(board, PLAYER1, np.int8(4))
    apply_player_action(board, np.int8(6), PLAYER2)
    assert connected_four(board, PLAYER2)
    assert connected_four(board, PLAYER2, np.int8(6))


def test_connected4_vertical():
    board = string_to_board(
        """
            |==============|
            |              |
            |              |
            |              |
            |  X           |
            |  X X X       |
            |  X X O O O   |
            |==============|
            |0 1 2 3 4 5 6 |
            """
    )
    assert not (connected_four(board, PLAYER1))
    apply_player_action(board, np.int8(1), PLAYER1)
    assert connected_four(board, PLAYER1)
    assert connected_four(board, PLAYER1, np.int8(1))


def test_connected4_diagonal_right():
    board = string_to_board(
        """
            |==============|
            |              |
            |              |
            |              |
            |  X   X O     |
            |  X X O X     |
            |  X X O O O   |
            |==============|
            |0 1 2 3 4 5 6 |
            """
    )
    assert not (connected_four(board, PLAYER1))
    apply_player_action(board, np.int8(4), PLAYER1)
    assert connected_four(board, PLAYER1)
    assert connected_four(board, PLAYER1, np.int8(4))


def test_connected4_diagonal_right2():
    board = string_to_board(
        """
            |==============|
            |              |
            |              |
            |        X     |
            |  X     O O X |
            |  X X X O X X |
            |  X X O O O X |
            |==============|
            |0 1 2 3 4 5 6 |
            """
    )
    assert not (connected_four(board, PLAYER2))
    apply_player_action(board, np.int8(6), PLAYER2)
    assert connected_four(board, PLAYER2)
    assert connected_four(board, PLAYER2, np.int8(6))


def test_connected4_diagonal_left():
    board = string_to_board(
        """
            |==============|
            |              |
            |              |
            |      X       |
            |    X O X     |
            |    X O O     |
            |    X O O O   |
            |==============|
            |0 1 2 3 4 5 6 |
            """
    )
    assert not (connected_four(board, PLAYER2))
    apply_player_action(board, np.int8(2), PLAYER2)
    assert connected_four(board, PLAYER2)
    assert connected_four(board, PLAYER2, np.int8(2))


def test_connected4_diagonal_left2():
    board = string_to_board(
        """
            |==============|
            |              |
            |              |
            |    O X       |
            |  X O O X     |
            |  O X O O X   |
            |  X X O X O O |
            |==============|
            |0 1 2 3 4 5 6 |
            """
    )
    assert not (connected_four(board, PLAYER1))
    apply_player_action(board, np.int8(2), PLAYER1)
    assert connected_four(board, PLAYER1)
    assert connected_four(board, PLAYER1, np.int8(2))


## check_end_state tests ##


def test_game_still_going_and_won():
    board = string_to_board(
        """
            |==============|
            |              |
            |              |
            |      O       |
            |  X O X X     |
            |  O X O O X   |
            |  X X O X O O |
            |==============|
            |0 1 2 3 4 5 6 |
            """
    )
    assert check_end_state(board, PLAYER1) == GameState.STILL_PLAYING
    apply_player_action(board, np.int8(4), PLAYER1)
    assert check_end_state(board, PLAYER1) == GameState.IS_WIN
    assert check_end_state(board, PLAYER1, np.int8(4)) == GameState.IS_WIN


def test_game_draw():
    board = string_to_board(
        """
            |==============|
            |X O X O X O X |
            |X O X O X O X |
            |X O X O X O X |
            |O X O X O X O |
            |O X O X O X O |
            |O X O X O X O |
            |==============|
            |0 1 2 3 4 5 6 |
            """
    )
    assert check_end_state(board, PLAYER1) == GameState.IS_DRAW


def test_game_still_going_and_lost():
    board = string_to_board(
        """
            |==============|
            |              |
            |              |
            |    O X       |
            |  X O X X     |
            |  O X O O X   |
            |  X X O X O O |
            |==============|
            |0 1 2 3 4 5 6 |
            """
    )
    assert check_end_state(board, PLAYER1) == GameState.STILL_PLAYING
    apply_player_action(board, np.int8(2), PLAYER1)
    assert check_end_state(board, PLAYER2) == GameState.IS_LOST
    assert check_end_state(board, PLAYER2, np.int8(2)) == GameState.IS_LOST


## if_game_ended tests ##


def test_if_game_ended_lost():
    board = string_to_board(
        """
            |==============|
            |              |
            |              |
            |    X O       |
            |  X O X X     |
            |  O X O O X   |
            |  X X O X O O |
            |==============|
            |0 1 2 3 4 5 6 |
            """
    )
    assert not if_game_ended(board)
    apply_player_action(board, np.int8(4), PLAYER1)
    assert if_game_ended(board)
    assert if_game_ended(board, np.int8(4))


def test_if_game_ended_lost2():
    board = string_to_board(
        """
            |==============|
            |              |
            |              |
            |    O X       |
            |  X O X X     |
            |  X O O O X   |
            |  X X O X O O |
            |==============|
            |0 1 2 3 4 5 6 |
            """
    )
    assert not if_game_ended(board)
    apply_player_action(board, np.int8(2), PLAYER2)
    assert if_game_ended(board)
    assert if_game_ended(board, np.int8(2))


def test_if_game_ended_draw():
    board = string_to_board(
        """
            |==============|
            |X O X O X O X |
            |X O X O X O X |
            |X O X O X O X |
            |O X O X O X O |
            |O X O X O X O |
            |O X O X O X O |
            |==============|
            |0 1 2 3 4 5 6 |
            """
    )
    assert if_game_ended(board)
