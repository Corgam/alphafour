import functools
import math

import numpy as np
from typing import Set


class Node:
    def __init__(self, game, move, parent):
        self.parent = parent
        self.game = game
        self.move = move
        self.children: Set[Node] = set()
        self.visitCount = 0
        self.score = 0

    def is_terminal(self):
        if len(self.children) == 0:
            return True
        return False

    def get_reward(self):
        # calculate reward here
        return 3

    def get_children(self) -> 'Node':
        # return set of all kids
        return {}

    def fully_expanded(self):
        pass


def choose(self, node):
    # choose best child node
    if node.is_terminal:
        print("don't choose terminal node man")


def calc_score(child, parent):
    exploitation_score = child.get_reward() / child.visitCount
    exploration_score = math.sqrt(4 * math.log(parent.visitCount) / child.visitCount)
    return exploitation_score + exploration_score


def get_best_move(node: Node):
    # returns child node with highest UCT value
    return functools.reduce(
        lambda acc, curr: acc if calc_score(acc, node) > calc_score(curr, node) else curr, node.children, node
    )
