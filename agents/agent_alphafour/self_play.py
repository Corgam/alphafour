from datetime import datetime
import os
import pickle

import numpy as np

from agents.agent_alphafour.mcts_with_NN import Connect4State, run_AlphaFour, Node
from agents.common import initialize_game_state, pretty_print_board
from agents.helpers import BoardPiece, PLAYER1, convert_number2print, GameState, get_rival_piece


def calculatePolicy(node: Node):
    """
    Calculates policy
    :param node:
    :return:
    """
    sumVisits = 0
    for child in node.children:
        sumVisits += child.visits
    policy = [child.visits / sumVisits for child in node.children]
    return policy


def save_into_file(filename, dataset_finished: list):
    full_path = os.path.join("agents/agent_alphafour/training_data/", filename)
    with open(full_path, "wb") as file:
        pickle.dump(dataset_finished, file)


def MCTS_self_play(board: np.ndarray = initialize_game_state(), player: BoardPiece = PLAYER1, iterations: int = 10):
    for iteration in range(iterations):
        # Init variables
        state = Connect4State(board, player)
        dataset_not_finished = []
        dataset_finished = []
        value = 0
        # Play the game
        while state.get_possible_moves():
            # TODO: Game does not end when there is end of the game, but lasts until the end. Should it be fixed?
            print(pretty_print_board(state.board))
            if state.player_just_moved == 1:
                move, root_node = run_AlphaFour(state, 1000)
            else:
                move, root_node = run_AlphaFour(state, 100)
            policy = calculatePolicy(root_node)
            dataset_not_finished.append([state.board.copy(), policy])
            # Make the move
            print("Move: " + str(move) + "\n")
            state.move(move)
        # Check who won
        if state.get_reward(state.player_just_moved) == GameState.IS_WIN:
            value = 1
            print("Player with symbol " + convert_number2print(state.player_just_moved) + " wins!")
        elif state.get_reward(state.player_just_moved) == GameState.IS_LOST:
            value = -1
            print("Player with symbol " + convert_number2print(get_rival_piece(state.player_just_moved)) + " wins!")
        else:
            print("Draw!")
        # Save data
        for i, data in enumerate(dataset_not_finished):
            board, policy = data
            if i == 0:
                dataset_finished.append([board, policy, 0])
            else:
                dataset_finished.append([board, policy, value])
        timeStr = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        save_into_file(f"data_iter{iteration}_" + timeStr + ".pkl", dataset_finished)
