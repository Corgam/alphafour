from __future__ import annotations  # Used for type hinting of class inside of itself

import random
from typing import List
import numpy as np

from agents.common import (
    apply_player_action,
    if_game_ended,
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


class Connect4StateMCTS:
    """
    Class for state of the connect four game.
    Holds information about the board and the last player who played.
    """

    def __init__(self, board=initialize_game_state(), player=PLAYER1):
        self.player_just_moved: BoardPiece = get_rival_piece(
            player
        )  # Player who moved last.
        self.board: np.ndarray = board  # The board itself

    def copy(self) -> Connect4StateMCTS:
        """
        Return a copy of the connect four state
        :return: copy_of_state
        """
        copy = Connect4StateMCTS()
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


class NodeMCTS:
    """
    A node class for MCTS. It is used to store information about:
    the state , move of the parent, parent node, list of child nodes, number of wins,
    number of visits, list of untried moves and which player just moved.
    """

    def __init__(
        self,
        state: Connect4StateMCTS = None,
        parent_move: PlayerAction = None,
        parent: NodeMCTS = None,
    ):
        self.parent_move: PlayerAction = parent_move  # Move which the parent carried out
        self.parent = parent  # Node of the parent. None if self is a root node.
        self.children: List[NodeMCTS] = []  # Set of all possible children
        self.visits: int = 0  # How many times this node has been visited
        self.wins: int = 0  # How many times this node has won
        self.untried_moves: list[
            PlayerAction
        ] = state.get_possible_moves()  # List of all moves possible from that node.
        self.player_just_moved = state.player_just_moved  # Player number who just moved

    def add_child(self, move: PlayerAction, state: Connect4StateMCTS) -> NodeMCTS:
        """
        Adds a child to the children list and deletes one move from list of untried moves.
        :param move: move which leads to the child
        :param state: connect 4 state
        :return: created_child
        """
        child = NodeMCTS(state, move, self)
        self.untried_moves.remove(move)
        self.children.append(child)
        return child

    def select_best_child(self) -> NodeMCTS:
        """
        Calculates the UCB1 for each of the children and returns the child with the biggest UCB1.
        https://cs.stackexchange.com/questions/113473/difference-between-ucb1-and-uct
        :return: child_with_biggest_UCB1
        """
        ucts = [
            child.wins / child.visits + np.sqrt(2 * np.log(self.visits) / child.visits)
            for child in self.children
        ]
        return self.children[np.argmax(ucts)]

    def backpropagate(self, result: GameState):
        """
        Used for backpropagation. Updates the number of visits and wins.
        :param result: the GameState to backpropagate
        """
        self.visits += 1
        if result == GameState.IS_WIN:
            self.wins += 1


def select_node_mcts(node: NodeMCTS, state: Connect4StateMCTS):
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


def expand_mcts(node: NodeMCTS, state: Connect4StateMCTS):
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


def rollout_mcts(state: Connect4StateMCTS):
    """
    Rollouts the state, until the game is ended.
    :param state: current state
    :return: state_after_rollout
    """
    while not if_game_ended(state.board):
        state.move(random.choice(state.get_possible_moves()))
    return state


def backpropagate_mcts(node: NodeMCTS, state: Connect4StateMCTS):
    """
    Backpropagates the value up the tree
    :param node: node to start backpropagation from
    :param state: current state
    """
    while node is not None:
        node.backpropagate(state.get_reward(node.player_just_moved))
        node = node.parent


def run_basic_mcts(
    root_state: Connect4StateMCTS, simulation_no=100
) -> (PlayerAction, NodeMCTS):
    """
    Runs MCTS simulation for a given root state n times.
    :param simulation_no: number of simulations to do
    :param root_state: beginning state
    :return: best_move
    """
    root_node = NodeMCTS(root_state)
    # Run simulation_no times the MCTS simulation
    for i in range(simulation_no):
        node = root_node
        state = root_state.copy()
        # 1. Select the node
        node, state = select_node_mcts(node, state)
        # 2. Expand the selected node
        node, state = expand_mcts(node, state)
        # 3. Rollout the selected node until the end of the game
        state = rollout_mcts(state)
        # 4. Backpropagate
        backpropagate_mcts(node, state)
    # Choose the child
    children_visits = [child.visits for child in root_node.children]
    return root_node.children[np.argmax(children_visits)].parent_move, root_node
