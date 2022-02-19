### CODE BY EMIL BALITZKI ###
from agents.common import initialize_game_state, string_to_board
from agents.helpers import (
    check_piece,
    get_rival_piece,
    convert_number2print,
    convert_print2number,
)
from agents.helpers import (
    NO_PLAYER,
    PLAYER1,
    PLAYER2,
    PLAYER1_PRINT,
    PLAYER2_PRINT,
    NO_PLAYER_PRINT,
)


def test_print_to_number():
    assert convert_print2number(NO_PLAYER_PRINT) == NO_PLAYER
    assert convert_print2number(PLAYER1_PRINT) == PLAYER1
    assert convert_print2number(PLAYER2_PRINT) == PLAYER2


def test_number_to_print():
    assert convert_number2print(NO_PLAYER) == NO_PLAYER_PRINT
    assert convert_number2print(PLAYER1) == PLAYER1_PRINT
    assert convert_number2print(PLAYER2) == PLAYER2_PRINT


def test_check_piece_simple():
    board = initialize_game_state()
    assert check_piece(board, (2, 2), NO_PLAYER)


def test_check_piece_complex():
    board = string_to_board(
        """
            |==============|
            |              |
            |              |
            |  O           |
            |  X   X O     |
            |  X X O X     |
            |  X X O O O   |
            |==============|
            |0 1 2 3 4 5 6 |
            """
    )
    assert check_piece(board, (2, 2), NO_PLAYER)
    assert check_piece(board, (2, 1), PLAYER2)
    assert check_piece(board, (3, 1), PLAYER1)
    assert check_piece(board, (5, 6), NO_PLAYER)
    assert check_piece(board, (3, 3), PLAYER1)


def test_get_rival():
    assert get_rival_piece(PLAYER1) == PLAYER2
    assert get_rival_piece(PLAYER2) == PLAYER1
