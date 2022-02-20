from __future__ import annotations  # Used for type hinting of class inside of itself

import os
import random
from typing import List
import numpy as np
import torch
from from_root import from_root

from agents.agent_alphafour.NN import AlphaNet
from agents.common import (
    apply_player_action,
    check_end_state,
    initialize_game_state,
    pretty_print_board,
)
from agents.helpers import (
    calculate_possible_moves,
    get_rival_piece,
    PlayerAction,
    GameState,
    PLAYER1,
    BoardPiece,
)


class Connect4State:
    """
    Class for state of the connect four game.
    Holds information about the board and the last player who played.
    """
    def __init__(self, board=initialize_game_state(), player=PLAYER1):
        self.player_just_moved: BoardPiece = get_rival_piece(player)  # Player who moved last.
        self.board: np.ndarray = board  # The board itself

    def copy(self) -> Connect4State:
        """
        Return a copy of the connect four state
        :return: copy_of_state
        """
        copy = Connect4State()
        copy.board = self.board.copy()
        copy.player_just_moved = self.player_just_moved
        return copy

    def move(self, move: PlayerAction):
        """
        Makes a specified move on the board. Player which move will be made is a rival of self.playerJustMoved
        :param move: move to do
        """
        apply_player_action(self.board, move, get_rival_piece(self.player_just_moved))
        self.player_just_moved = get_rival_piece(self.player_just_moved)

    def get_possible_moves(self) -> list[PlayerAction]:
        """
        Returns a list of possible moves (not full columns) from the board.
        :return: list_of_possible_moves
        """
        return calculate_possible_moves(self.board)

    def get_reward(self, player: BoardPiece) -> GameState:
        """
        Returns a GameState object, symbolizing if given player has won or not.
        :return: end_game_state
        """
        return check_end_state(self.board, player)

    def __repr__(self) -> str:
        """
        String representation of the Connect 4 State
        :return:
        """
        return pretty_print_board(self.board)


class Node:
    """
    A node class for MCTS. It is used to store information about:
    the state , move of the parent, parent node, list of child nodes, number of wins,
    number of visits, list of untried moves and which player just moved.
    """
    def __init__(
        self,
        state: Connect4State = None,
        parent_move: PlayerAction = None,
        parent: Node = None,
    ):
        self.parent_move: PlayerAction = parent_move  # Move which the parent carried out
        self.parent: Node = parent  # Node of the parent. None if self is a root node.
        self.children: List[Node] = []  # Set of all possible children
        self.visits: int = 0  # How many times this node has been visited
        self.wins: float = 0  # How many times this node has won
        self.untried_moves: list[PlayerAction] = state.get_possible_moves()  # List of all moves possible
        self.state: Connect4State = state
        self.children_priorities: np.ndarray = np.zeros([7], np.float32)

    def add_child(self, move: PlayerAction, state: Connect4State) -> Node:
        """
        Adds a child to the children list and deletes one move from list of untried moves.
        :param move: move which leads to the child
        :param state: connect 4 state
        :return: created_child
        """
        child = Node(state, move, self)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def select_best_child(self) -> Node:
        """
        Calculates the UCB1 for each of the children and returns the child with the biggest UCB1.
        https://cs.stackexchange.com/questions/113473/difference-between-ucb1-and-uct
        :return: child_with_biggest_UCB1
        """
        ucts = [
            child.wins / child.visits
            + np.sqrt(
                2
                * np.log(self.visits)
                * (
                    abs(self.children_priorities[self.children.index(child)])
                    / child.visits
                )
            )
            for child in self.children
        ]
        return self.children[np.argmax(ucts)]

    def backpropagate(self, result: float):
        """
        Used for backpropagation. Updates the number of visits and wins.
        :param result: the value to backpropagate
        """
        self.visits += 1
        self.wins += result


def select_node(node: Node, state: Connect4State):
    """
    Selects the node for expansion.
    NOTE: This function rises Duplicated Code warning, but because we do not want to mix both agents (pure MCTS and
    AlphaFour) we leave this warning out.
    :param state: current state
    :param node: current node
    :return: selected_node, new_state
    """
    while node.untried_moves == [] and node.children != []:
        node = node.select_best_child()
        state.move(node.parent_move)
    return node, state


def expand(node: Node, state: Connect4State):
    """
    Expands the given node.
    :param node: node_to_expand
    :param state: current state
    :return: child_node, new_state
    """
    if node.untried_moves:
        move = random.choice(node.untried_moves)
        state.move(move)
        node = node.add_child(move, state)
    return node, state


def rollout(node: Node, child_priorities: np.ndarray):
    """
    Rollouts the state, until the game is ended.
    :param child_priorities: the priorities of children taken from NN
    :param node: starting node
    :return: state_after_rollout
    """
    moves = node.untried_moves
    temp_priorities = child_priorities
    # If it is the root node, make the priorities a little bit random, so each run is different
    if node.parent is None:
        random_values = np.random.dirichlet(
            np.zeros([len(temp_priorities)], np.float32) + 192
        )
        for prior in range(len(temp_priorities)):
            temp_priorities[prior] = (
                0.75 * temp_priorities[prior] + 0.25 * random_values[prior]
            )
    # Mask all unavailable moves
    for i in range(len(child_priorities)):
        if i not in moves:
            temp_priorities[i] = 0
    node.children_priorities = temp_priorities
    return node.state


def backpropagate(node: Node, state: Connect4State, value_estimate: float):
    """
    Backpropagates the value up the tree
    :param value_estimate: the value estimate taken from NN
    :param node: node to start backpropagation from
    :param state: current state
    """
    while node is not None:
        if state.player_just_moved == PLAYER1:
            node.backpropagate(value_estimate)
        else:
            node.backpropagate(-1 * value_estimate)
        node = node.parent


def get_nn_outputs(nn: AlphaNet, node: Node):
    """
    Gets the predicted value estimate and the child priorities from the NN.
    Make sures the board is transformed into right dimensions for NN to accept it.
    :param nn: the Neural Network which will give back the estimates.
    :param node: the node containing board state
    :return: child_priorities, value_estimate
    """
    # Prepare the board
    board = node.state.board
    # Change the dimensions of the tensor to 4D
    board = np.expand_dims(board, 0)  # Dims (1,6,7)
    board = np.expand_dims(board, 1)  # Dims (1,1,6,7)
    # Change the type of the tensor
    board = torch.from_numpy(board).float()
    # Get values from the NN
    child_priorities, value_estimate = nn(board)
    # Unpack the values
    child_priorities = (
        child_priorities.detach().cpu().numpy()
    )  # Change to numpy array
    child_priorities = child_priorities.reshape(-1)  # Delete one dimension
    value_estimate = value_estimate.item()
    return child_priorities, value_estimate


def run_single_mcts(root_state: Connect4State, simulation_no: int, nn: AlphaNet) -> (PlayerAction, Node):
    """
    Runs MCTS simulation for a given root state n times.
    :param simulation_no: number of simulations to do
    :param root_state: beginning state
    :param nn: the neural network
    :return: best_move
    """
    # Create the root node
    root_node = Node(root_state)
    # Run simulation_no times the MCTS simulation
    for i in range(simulation_no):
        node = root_node
        state = root_state.copy()
        # 1. Select the node
        node, state = select_node(node, state)
        # 2. Expand the selected node
        node, state = expand(node, state)
        # 3. Ask the NN for values
        child_priorities, value_estimate = get_nn_outputs(nn, node)
        # 4. Rollout the selected node until the end of the game
        state = rollout(node, child_priorities)
        # 5. Backpropagate
        backpropagate(node, state, value_estimate)
    # Choose the child
    children_visits = [child.visits for child in root_node.children]
    return root_node.children[np.argmax(children_visits)].parent_move, root_node


def run_alpha_four(root_state: Connect4State, simulation_no: int, nn_iteration: int):
    """
    The main function to run the MCTS on the root state.
    Creates or loads up the NN before simulation.
    :param root_state: the root state to start simulation from
    :param simulation_no: how many simulations of MCTS to do
    :param nn_iteration: the iteration of NN to load
    :return:
    """
    # Create the NN
    nn = AlphaNet()
    nn.eval()  # Turn on the evaluation mode
    # Load the NN if provided
    nn_filename = from_root(
        f"agents/agent_alphafour/trained_NN/NN_iteration{nn_iteration}.pth.tar"
    )
    if os.path.isfile(nn_filename):
        loaded_nn = torch.load(nn_filename)
        nn.load_state_dict(loaded_nn["state_dict"])
    else:
        torch.save({"state_dict": nn.state_dict()}, nn_filename)
    with torch.no_grad():
        move, root_node = run_single_mcts(root_state, simulation_no, nn)
    return move, root_node
