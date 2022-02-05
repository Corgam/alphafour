from __future__ import annotations  # Used for type hinting of class inside of itself

import os
import random
from typing import List
import numpy as np
import torch.cuda
from from_root import from_root
from agents.agent_alphafour.neural import AlphaNet
from agents.common import apply_player_action, if_game_ended, check_end_state, initialize_game_state, pretty_print_board
from agents.helpers import calculate_possible_moves, get_rival_piece, PlayerAction, GameState, PLAYER1, BoardPiece


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
        :return:
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
        :return:
        """
        return check_end_state(self.board, player)

    def __repr__(self):
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

    def __init__(self, state: Connect4State = None, parent_move: PlayerAction = None, parent: Node = None):
        self.parent_move: PlayerAction = parent_move  # Move which the parent carried out
        self.parent = parent  # Node of the parent. None if self is a root node.
        self.children: List[Node] = []  # Set of all possible children
        self.visits: int = 0  # How many times this node has been visited
        self.wins: int = 0  # How many times this node has won
        self.untried_moves: list[
            PlayerAction] = state.get_possible_moves()  # List of all moves possible from that node.
        self.state = state
        self.children_priorities = np.zeros([7], np.float32)

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
        # UCTs = [child.wins / child.visits + np.sqrt(2 * np.log(self.visits) / child.visits) for child in /
        # self.children]
        # TODO: Check if the calculations are right
        # TODO: Maybe replace the first value?
        UCTs = [child.wins / child.visits + np.sqrt(
            2 * np.log(self.visits) * (abs(self.children_priorities[self.children.index(child)]) / child.visits)) for
                child in self.children]
        return self.children[np.argmax(UCTs)]

    def backpropagate(self, result: GameState):
        """
        Used for backpropagation. Updates the number of visits and wins.
        :param result: the GameState to backpropagate
        """
        self.visits += 1
        if result == GameState.IS_WIN:
            self.wins += 1


def select_node(node: Node, state: Connect4State):
    """
    Selects the node for expansion.
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
    Expands the given node
    :param node: node_to_expand
    :param state: current state
    :return: child_node, new_state
    """
    if node.untried_moves:
        move = random.choice(node.untried_moves)
        state.move(move)
        node = node.add_child(move, state)
    return node, state


def rollout(node, child_priorities):
    """
    Rollouts the state, until the game is ended.
    :param child_priorities:
    :param node:
    :return: state_after_rollout
    """
    moves = node.untried_moves
    temp_priorities = child_priorities
    # If it is the root node, make the priorities a little bit random, so each run is different
    if node.parent is None:
        randomValues = np.random.dirichlet(np.zeros([len(temp_priorities)], np.float32) + 192)
        for prior in range(len(temp_priorities)):
            temp_priorities[prior] = 0.75 * temp_priorities[prior] + 0.25 * randomValues[prior]
    # Mask all unavailable moves
    for i in range(len(child_priorities)):
        if i not in moves:
            temp_priorities[i] = 0
    node.children_priorities = temp_priorities
    return node.state
    # while not if_game_ended(state.board):
    #     state.move(random.choice(state.get_possible_moves()))
    # return state


def backpropagate(node: Node, state: Connect4State, value_estimate):
    """
    Backpropagates the value up the tree
    :param value_estimate:
    :param node: node to start backpropagation from
    :param state: current state
    """
    while node is not None:
        # TODO: Check if the value is negative for correct player.
        if state.player_just_moved == PLAYER1:
            node.backpropagate(value_estimate)
        else:
            node.backpropagate(-1 * value_estimate)
        node = node.parent


def get_NN_outputs(NN: AlphaNet, node: Node):
    """

    :param NN:
    :param node:
    :return:
    """
    # TODO: Implement the real NN
    child_priorities = [0.2, 0.1, 0.3, 0.18, 0.11, 0.01, 0.5]
    value_estimate = 0.4
    # child_priorities, value_estimate = NN(node.state.board)
    return child_priorities, value_estimate


def run_single_MCTS(root_state: Connect4State, simulation_no: int, NN: AlphaNet) -> (PlayerAction, Node):
    """
    Runs MCTS simulation for a given root state n times.
    :param simulation_no: number of simulations to do
    :param root_state: beginning state
    :param NN: the neural network
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
        child_priorities, value_estimate = get_NN_outputs(NN, node)
        # 4. Rollout the selected node until the end of the game
        state = rollout(node, child_priorities)
        # 5. Backpropagate
        backpropagate(node, state, value_estimate)
    # Choose the child
    children_visits = [child.visits for child in root_node.children]
    return root_node.children[np.argmax(children_visits)].parent_move, root_node


def run_AlphaFour(root_state: Connect4State, simulation_no=100, NN_iteration=0):
    # Create the NN
    NN = AlphaNet()
    NN.eval()  # Turn on the evaluation mode
    # Load the NN if provided
    NN_filename = from_root(f"agents/agent_alphafour/trained_NN/NN_iteration{NN_iteration}.pth.tar")
    if os.path.isfile(NN_filename):
        loaded_NN = torch.load(NN_filename)
        NN.load_state_dict(loaded_NN["state_dict"])
        print(f"Loaded {NN_filename} neural network.")
    else:
        torch.save({"state_dict": NN.state_dict()}, NN_filename)
        print(f"Created new {NN_filename} neural network.")
    # Turn on CUDA
    if torch.cuda.is_available():
        NN.cuda()
    print("Started MCTS...")
    with torch.no_grad():
        run_single_MCTS(root_state, simulation_no, NN)
    print("MCTS has finished!")
