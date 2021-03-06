import os
import pickle

import numpy as np
from typing import Optional, Callable

from from_root import from_root

from agents.agent_MCTS.gen_move import generate_move_mcts
from agents.agent_alphafour.self_play import mcts_self_play
from agents.agent_alphafour.evaluator import evaluate_nn
from agents.agent_alphafour.trainNN import train_nn
from agents.common import PlayerAction, BoardPiece, initialize_game_state
from agents.helpers import SavedState, GenMove, PLAYER1
from agents.agent_alphafour import generate_move_alphafour


# Read the keyboard input for user move
def user_move(
        board: np.ndarray, _player: BoardPiece, saved_state: Optional[SavedState]
):
    action = PlayerAction(-1)
    while not 0 <= action < board.shape[1]:
        try:
            action = PlayerAction(input("Column? "))
        except ValueError:
            print(
                "Input could not be converted to the dtype PlayerAction, try entering an integer."
            )
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
    from agents.common import (
        initialize_game_state,
        pretty_print_board,
        apply_player_action,
        check_end_state,
    )

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
                    f"{player_name} you are playing with {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}"
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
                            f"{player_name} won playing {PLAYER1_PRINT if player == PLAYER1 else PLAYER2_PRINT}"
                        )
                    # Stop playing
                    playing = False
                    break


NUMBER_OF_ITERATIONS = 5
NUMBER_OF_MCTS_GAMES_PER_ITERATION = 10
NUMBER_OF_GAMES_PER_EVALUATION = 10
NUMBER_OF_TRAINING_EPOCHS = 10
NUMBER_OF_MCTS_SIMULATIONS = 100


def main_pipeline():
    """
    Runs the main pipeline of AlphaFour, given number of times.
    """
    # Create folders
    if not os.path.exists(f"agents/agent_alphafour/trained_NN/"):
        os.makedirs(f"agents/agent_alphafour/trained_NN/")
    if not os.path.exists(f"agents/agent_alphafour/training_data/"):
        os.makedirs(f"agents/agent_alphafour/training_data/")
    if not os.path.exists(f"agents/agent_alphafour/evaluation_data/"):
        os.makedirs(f"agents/agent_alphafour/evaluation_data/")
    # Run the iterations
    for iteration in range(NUMBER_OF_ITERATIONS):
        print(f"[PIPELINE] Started {iteration} iteration.")
        # Run the self-play MCTS and generate the data
        mcts_self_play(
            iteration=iteration,
            board=initialize_game_state(),
            player=PLAYER1,
            number_of_mcts_simulations=NUMBER_OF_MCTS_SIMULATIONS,
            number_of_games=NUMBER_OF_MCTS_GAMES_PER_ITERATION,
        )
        # Train the NN with data from MCTS
        train_nn(iteration=iteration, num_of_epochs=NUMBER_OF_TRAINING_EPOCHS)
        # Starting from second iteration, evaluate the NNs and choose the better one.
        if iteration > 0:
            better_nn = evaluate_nn(
                iteration, iteration + 1, number_of_games=NUMBER_OF_GAMES_PER_EVALUATION,
                number_of_mcts_simulations=NUMBER_OF_MCTS_SIMULATIONS
            )
            additional_runs = 0
            # If the new NN is not good enough, train it more until it is.
            while better_nn != (iteration + 1):
                print(
                    "[PIPELINE] New NN not strong enough! More training needs to be done..."
                )
                mcts_self_play(
                    iteration=iteration,
                    board=initialize_game_state(),
                    number_of_games=NUMBER_OF_MCTS_GAMES_PER_ITERATION,
                    number_of_mcts_simulations=NUMBER_OF_MCTS_SIMULATIONS,
                    player=PLAYER1,
                )
                train_nn(iteration=iteration, num_of_epochs=NUMBER_OF_TRAINING_EPOCHS)
                better_nn = evaluate_nn(
                    iteration,
                    iteration + 1,
                    number_of_games=NUMBER_OF_GAMES_PER_EVALUATION,
                    number_of_mcts_simulations=NUMBER_OF_MCTS_SIMULATIONS
                )
                additional_runs += 1


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
        # Select the iteration
        print("\n")
        data_path = f"agents/agent_alphafour/trained_NN/"
        number_of_iterations = len(os.listdir(data_path))
        if number_of_iterations == 0:
            print("No trained agent available! Firstly, train the agent yourself!")
        else:
            print(
                f"Choose the iteration of the AlphaFour from 0 to {number_of_iterations - 1}"
            )
            it = input("Chosen iteration:")
            filePath = from_root("chosen_iteration.pkl")
            with open(filePath, "wb") as f:
                pickle.dump({"iteration": it, "number_of_mcts_simulations": NUMBER_OF_MCTS_SIMULATIONS}, f)
            human_vs_agent(generate_move_alphafour)
    elif agent == "4":
        human_vs_agent(generate_move_mcts)
    else:
        print("Wrong number selected. Restart program and select number from 1 to 4.")
