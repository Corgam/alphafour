from datetime import datetime
import os
import pickle

import numpy as np

from agents.agent_alphafour.mcts_with_NN import Connect4State, run_alpha_four, Node
from agents.common import if_game_ended
from agents.helpers import BoardPiece, GameState


def calculate_policy(node: Node) -> np.ndarray:
    """
    Calculates the policy for a given node.
    :param node: node for which the policy will be calculated
    :return: the policy
    """
    sum_visits = 0
    policy = np.zeros([7], np.float32)
    # Calculate the overall number of visits.
    for child in node.children:
        sum_visits += child.visits
    # Calculate all values (for full columns, leave 0)
    for child in node.children:
        policy[child.parent_move] = child.visits / sum_visits
    return policy


def save_file(filename: str, dataset_finished: list, iteration: int):
    """
    Saves finished dataset to file in agents/agent_alphafour/training_data/<iteration>/<filename>
    :param filename: file name
    :param dataset_finished: final dataset to save on the disk
    :param iteration: current iteration of the main pipeline
    """
    if not os.path.exists(
        f"agents/agent_alphafour/training_data/iteration{iteration}/"
    ):
        os.makedirs(f"agents/agent_alphafour/training_data/iteration{iteration}/")
    full_path = os.path.join(
        f"agents/agent_alphafour/training_data/iteration{iteration}/", filename
    )
    with open(full_path, "wb") as file:
        pickle.dump(dataset_finished, file)


def mcts_self_play(
    iteration: int,
    board: np.ndarray,
    player: BoardPiece,
    number_of_mcts_simulations: int,
    number_of_games: int,
):
    """
    Runs MCTS-based self play between two MCTS agents.
    Saves generated data on the drive.
    :param iteration: iteration of the main pipeline
    :param board: the board to start game from
    :param player: the next player to move
    :param number_of_mcts_simulations: number of MCTS simulations
    :param number_of_games: number of self play games to do
    """
    print("[MCTS] Started MCTS plays!")
    starting_state = Connect4State(board, player)
    for game in range(number_of_games):
        print(f"[MCTS] Started playing MCTS game number: {game}")
        # Init variables
        state = starting_state.copy()
        dataset_not_finished = []
        dataset_finished = []
        value = 0
        # Play the game
        while state.get_possible_moves() and if_game_ended(state.board) is False:
            move, root_node = run_alpha_four(state, number_of_mcts_simulations, iteration)
            policy = calculate_policy(root_node)
            dataset_not_finished.append([state.board.copy(), policy])
            # Make the move
            state.move(move)
        # Check who won
        if state.get_reward(state.player_just_moved) == GameState.IS_WIN:
            value = 1
        elif state.get_reward(state.player_just_moved) == GameState.IS_LOST:
            value = -1
        else:
            print("Draw!")
        # Save data
        for i, data in enumerate(dataset_not_finished):
            loaded_board, loaded_policy = data
            if i == 0:
                dataset_finished.append([loaded_board, loaded_policy, 0])
            else:
                dataset_finished.append([loaded_board, loaded_policy, value])
        time_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
        print(f"[MCTS] Finished playing MCTS game number: {game}. Saving results...")
        save_file(f"data_game{game}_" + time_str + ".pkl", dataset_finished, iteration)
