import random
from typing import List
import numpy as np

from agents.common import apply_player_action, if_game_ended, check_end_state
from agents.helpers import calculate_possible_moves, get_rival_piece, PlayerAction, GameState

NOF_SIMULATIONS = 100  # Number of simulations to run


class Node:
    def __init__(self, board, next_player, parent=None, parent_move=None):
        self.board: np.ndarray = board  # State of connect 4 board
        self.parent: Node = parent  # Parent node
        self.next_player = next_player  # The player who will move next
        self.parent_move: PlayerAction = parent_move  # Move which the parent carried out
        self.children: List[Node] = []  # Set of all possible children
        self.visit_count: int = 0  # How many times this node has been visited
        self.wins: int = 0  #
        self.losses: int = 0  #
        # List of all moves which were not expanded.
        self.untriedMoves: list[PlayerAction] = calculate_possible_moves(board)

    def is_terminal(self):
        """
        Return if the node is a leaf, terminal node with finished game as a board state.
        :return: if_terminal
        """
        return if_game_ended(self.board)

    def expand(self):
        """
        Expands the node once, if not fully expanded.
        """
        if not self.is_fully_expanded():
            move = self.untriedMoves.pop(0)  # Get the first move from the list
            next_board = apply_player_action(self.board, move, self.next_player, True)
            new_child = Node(next_board, get_rival_piece(self.next_player), self, move)
            self.children.append(new_child)
            return new_child
        else:
            return None

    def is_fully_expanded(self):
        """
        Returns if the node has been fully expanded.
        :return: if_fully_expanded
        """
        return len(self.untriedMoves) == 0

    def rollout(self) -> GameState:
        """
        Plays the board till there is an outcome for the game.
        Returns the game state of the final board.
        :return:
        """
        board = self.board.copy()
        nextPlayer = self.next_player
        while not if_game_ended(board):
            possible_moves = calculate_possible_moves(board)
            selected_move = rollout_policy(possible_moves)
            apply_player_action(board, selected_move, nextPlayer)
            nextPlayer = get_rival_piece(nextPlayer)
        return check_end_state(board, self.next_player)

    def backpropagate(self, reward):
        """
        Backpropagates the result from the node itself up the the root.
        :param reward: the result to backpropagate
        """
        self.visit_count += 1
        # Add the result to the appropriate counter
        if reward == GameState.IS_WIN:
            self.wins += 1
        elif reward == GameState.IS_LOST:
            self.losses += 1
        # Call it in each parent, till at the root.
        if self.parent is not None:
            self.parent.backpropagate(reward)

    def get_best_child(self, exploration_param=0.1):
        """
        Returns child node with highest UCT value
        :return:
        """
        childrenUCT = [((child.wins - child.losses) / child.visit_count) + exploration_param * np.sqrt(
            2 * np.log(self.visit_count) / child.visit_count) for child in self.children]
        bestChildID = np.argmax(childrenUCT)
        return self.children[bestChildID]


def tree_policy(node: Node) -> Node:
    """
    Selects the node to run rollout.
    :param node:
    :return:
    """
    while not node.is_terminal():
        if not node.is_fully_expanded():
            return node.expand()
        else:
            return node.get_best_child()


def rollout_policy(possible_moves: list[PlayerAction]):
    """
    Selects the next move from possible moves based on the rollout policy.
    For now, it selects it randomly, preferring columns closer to the middle.
    TODO: Implement neural Network choice
    :param possible_moves:
    :return:
    """
    assert len(possible_moves) != 0
    # Generate the temporary list, starting from the middle
    columns_order = [3]
    # Then other columns in the order of how far they are from the middle (randomness for the same distance)
    for distance in [1, 2, 3]:
        temp_order = [3 - distance, 3 + distance]
        random.shuffle(temp_order)
        columns_order.extend(temp_order)
    # Select the column which is the closest to the middle
    for column_number in columns_order:
        if possible_moves.count(np.int8(column_number)):
            return np.int8(column_number)


def select_the_best_move(rootNode: Node):
    """
    Selects the best action for the given root node.
    :param rootNode: the root node
    :return: best_action
    """
    # Run the simulation many times to improve the outcome
    for i in range(NOF_SIMULATIONS):
        # 1. Select the node
        v = tree_policy(rootNode)
        # 2. Rollout the node (play until the game has finished)
        reward = v.rollout()
        # 3. Backpropagate the result up the tree
        v.backpropagate(reward)
    # After all simulations, return the best possibility for a move.
    return rootNode.get_best_child().parent_move
