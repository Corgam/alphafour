from datetime import datetime
import os
import pickle

import numpy as np

from agents.agent_alphafour.mcts_with_NN import Connect4State, run_AlphaFour, Node
from agents.common import initialize_game_state, pretty_print_board, if_game_ended
from agents.helpers import BoardPiece, PLAYER1, convert_number2print, GameState, get_rival_piece


def calculatePolicy(node: Node):
    """
    Calculates policy
    :param node:
    :return:
    """
    sumVisits = 0
    policy = np.zeros([7], np.float32)
    # Calculate the overall number of visits.
    for child in node.children:
        sumVisits += child.visits
    # Calculate all values (for full columns, leave 0)
    for child in node.children:
        policy[child.parent_move] = child.visits / sumVisits
    return policy


def save_file(filename, dataset_finished: list, iteration):
    if not os.path.exists(f"agents/agent_alphafour/training_data/iteration{iteration}/"):
        os.makedirs(f"agents/agent_alphafour/training_data/iteration{iteration}/")
    full_path = os.path.join(f"agents/agent_alphafour/training_data/iteration{iteration}/", filename)
    with open(full_path, "wb") as file:
        pickle.dump(dataset_finished, file)


def MCTS_self_play(iteration, board: np.ndarray = initialize_game_state(), player: BoardPiece = PLAYER1,
                   number_of_games: int = 10, start_iter: int = 0):
    starting_state = Connect4State(board, player)
    for game in range(number_of_games):
        print(f"Started playing MCTS game number: {game}")
        # Init variables
        state = starting_state.copy()
        dataset_not_finished = []
        dataset_finished = []
        value = 0
        # Play the game
        while state.get_possible_moves() and if_game_ended(state.board) is False:
            #  print(pretty_print_board(state.board))
            if state.player_just_moved == 1:
                move, root_node = run_AlphaFour(state, 100)
            else:
                move, root_node = run_AlphaFour(state, 100)
            policy = calculatePolicy(root_node)
            dataset_not_finished.append([state.board.copy(), policy])
            # Make the move
            #  print("Move: " + str(move) + "\n")
            state.move(move)
        # Check who won
        if state.get_reward(state.player_just_moved) == GameState.IS_WIN:
            value = 1
            #  print("Player with symbol " + convert_number2print(state.player_just_moved) + " wins!")
        elif state.get_reward(state.player_just_moved) == GameState.IS_LOST:
            value = -1
            #  print("Player with symbol " + convert_number2print(get_rival_piece(state.player_just_moved)) + " wins!")
        else:
            print("Draw!")
        # Save data
        for i, data in enumerate(dataset_not_finished):
            loaded_board, loaded_policy = data
            if i == 0:
                dataset_finished.append([loaded_board, loaded_policy, 0])
            else:
                dataset_finished.append([loaded_board, loaded_policy, value])
        timeStr = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        print(f"Finished playing MCTS game number: {game}. Saving results...")
        save_file(f"data_game{game}_" + timeStr + ".pkl", dataset_finished, iteration)
