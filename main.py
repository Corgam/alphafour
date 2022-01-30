import numpy as np
from typing import Optional, Callable

from agents.agent_MCTS.gen_move import generate_move_MCTS
from agents.agent_alphafour.self_play import MCTS_self_play
from agents.agent_alphafour.evaluator import evaluate_NN
from agents.agent_alphafour.neural import trainNN
from agents.common import PlayerAction, BoardPiece
from agents.helpers import SavedState, GenMove
from agents.agent_alphafour import generate_move_alphafour


# Read the keyboard input for user move
def user_move(board: np.ndarray, _player: BoardPiece, saved_state: Optional[SavedState]):
    action = PlayerAction(-1)
    while not 0 <= action < board.shape[1]:
        try:
            action = PlayerAction(input("Column? "))
        except ValueError:
            print("Input could not be converted to the dtype PlayerAction, try entering an integer.")
    return action, saved_state


def human_vs_agent(
        generate_move_1: GenMove,  # First player - provided in the function
        generate_move_2: GenMove = user_move,  # Second player - always human
        player_1: str = "Player 1",  # Name of the first player
        player_2: str = "Player 2",  # Name of the second player
        args_1: tuple = (),
        args_2: tuple = (),
        init_1: Callable = lambda board, player: None,  # Initialization function for the first game
        init_2: Callable = lambda board, player: None,  # Initialization function for the second game
):
    import time
    from agents.helpers import PLAYER1, PLAYER2, PLAYER1_PRINT, PLAYER2_PRINT, GameState
    from agents.common import initialize_game_state, pretty_print_board, apply_player_action, check_end_state
    # Tuple containing the values for both players
    players = (PLAYER1, PLAYER2)
    for play_first in (1, -1):
        # Run the init functions for two players?
        for init, player in zip((init_1, init_2)[::play_first], players):
            init(initialize_game_state(), player)
        saved_state = {PLAYER1: None, PLAYER2: None}
        # Main board for the game
        board = initialize_game_state()
        # Tuple with functions generating moves
        gen_moves = (generate_move_1, generate_move_2)[::play_first]
        # Tuple with player names
        player_names = (player_1, player_2)[::play_first]
        # Tuple with moves generation arguments
        gen_args = (args_1, args_2)[::play_first]
        # Main play loop
        playing = True
        while playing:
            # For both players make a move, one after other, starting with first one.
            for player, player_name, gen_move, args in zip(
                    players, player_names, gen_moves, gen_args,
            ):
                # Save initial time
                t0 = time.time()
                # Print the starting info
                print(pretty_print_board(board))
                print(
                    f'{player_name} you are playing with {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}'
                )
                # Generate the move for the player
                action, saved_state[player] = gen_move(
                    board.copy(), player, saved_state[player], *args
                )
                # Check if there was an error with generating the move
                assert action != -1
                print(f"Move time: {time.time() - t0:.3f}s")
                # Make the actual move
                apply_player_action(board, action, player)
                # Get the end state
                end_state = check_end_state(board, player)
                # Check if the game is ended
                if end_state != GameState.STILL_PLAYING:
                    print(pretty_print_board(board))
                    # Check for draw
                    if end_state == GameState.IS_DRAW:
                        print("Game ended in draw")
                    # Check for the win
                    else:
                        print(
                            f'{player_name} won playing {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}'
                        )
                    # Stop playing
                    playing = False
                    break


def main_pipeline(iterations=100):
    """
    Runs the main pipeline of AlphaFour
    :param iterations:
    :return:
    """
    for i in range(iterations):
        # Run the self-play MCTS and generate the data
        MCTS_self_play()
        # Train the NN with data from MCTS
        trainNN()
        if i > 0:
            better_NN = evaluate_NN()
            # TODO: If the new NN does not win, train it more.


if __name__ == "__main__":
    print("Welcome to AlphaFour!")
    print("1. Run AlphaFour training")
    print("2. Play Human vs. Human")
    print("3. Play Human vs. AlphaFour Agent")
    print("4. Play Human vs. MCTS Agent")
    agent = input("Please type the number to choose the agent:")
    if agent == "1":
        main_pipeline()
    elif agent == "2":
        human_vs_agent(user_move)
    elif agent == "3":
        human_vs_agent(generate_move_alphafour)
    elif agent == "4":
        human_vs_agent(generate_move_MCTS)
    else:
        print("Wrong number selected. Restart program and select number from 1 to 4.")
