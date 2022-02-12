import os
import pickle

import torch
from from_root import from_root

from agents.agent_alphafour.NN import Alpha_Net
from agents.agent_alphafour.mcts_with_NN import Connect4State, run_AlphaFour, run_single_MCTS
from agents.common import initialize_game_state, if_game_ended
from agents.helpers import PLAYER1, GameState


def save_file(filename, data):
    if not os.path.exists(f"agents/agent_alphafour/evaluation_data/"):
        os.makedirs(f"agents/agent_alphafour/evaluation_data/")
    full_path = os.path.join(f"agents/agent_alphafour/evaluation_data/", filename)
    with open(full_path, "wb") as file:
        pickle.dump(data, file)


def load_file(filename):
    if os.path.exists(f"agents/agent_alphafour/evaluation_data/"):
        full_path = os.path.join(f"agents/agent_alphafour/evaluation_data/", filename)
        with open(full_path, "wb") as file:
            data = pickle.load(file)
        return data


class Match:
    def __init__(self, current_NN, best_NN):
        self.current_NN = current_NN
        self.best_NN = best_NN
        pass

    def solve(self, number_of_games: int):
        print("Starting NN Match!")
        wins = 0
        for i in range(number_of_games):
            with torch.no_grad():
                winner = self.play_round()
            if winner == "current":
                wins += 1
        save_file("wins_ratio", {"ratio": wins / number_of_games, "number_of_games": number_of_games})
        print("Finished NN Match!")

    def play_round(self):
        # TODO: Randomize the player
        state = Connect4State(initialize_game_state(), PLAYER1)
        starting_player = state.player_just_moved
        first_player_NN = self.current_NN
        second_player_NN = self.best_NN
        first_player = "current"
        second_player = "best"
        # Play the game
        while state.get_possible_moves() and if_game_ended(state.board) is False:
            #  print(pretty_print_board(state.board))
            if state.player_just_moved == 1:
                move, root_node = run_single_MCTS(state, 100, first_player_NN)
            else:
                move, root_node = run_single_MCTS(state, 100, second_player_NN)
            # Make the move
            state.move(move)
        # Check who won
        if state.get_reward(starting_player) == GameState.IS_WIN:
            return first_player
        elif state.get_reward(starting_player) == GameState.IS_LOST:
            return second_player
        else:
            # Draw
            return None


def evaluate_NN(best_NN, current_NN, number_of_games):
    print("Started evaluating the NNs.")
    # Prepare filenames for NNs
    best_NN_filename = from_root(f"agents/agent_alphafour/trained_NN/NN_iteration{best_NN}.pth.tar")
    current_NN_filename = from_root(f"agents/agent_alphafour/trained_NN/NN_iteration{current_NN}.pth.tar")
    best_NN = Alpha_Net()
    current_NN = Alpha_Net()
    best_NN.eval()
    current_NN.eval()
    # Load the current NN
    loaded_NN = torch.load(current_NN_filename)
    current_NN.load_state_dict(loaded_NN["state_dict"])
    loaded_NN = torch.load(best_NN_filename)
    best_NN.load_state_dict(loaded_NN["state_dict"])
    # Play the match
    match = Match(current_NN=current_NN, best_NN=best_NN)
    Match.solve(number_of_games)
    ratio = load_file("wins_ratio")
    if ratio.ratio >= 0.55:
        return current_NN
    else:
        return best_NN
