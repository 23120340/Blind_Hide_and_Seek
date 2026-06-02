"""
Example student submission showing the required interface.

Students should implement their own PacmanAgent and/or GhostAgent
following this template.
"""

"""
Script testing
python arena.py --seek thuan --hide example_student --pacman-speed 2 --capture-distance 2 --pacman-obs-radius 5 --ghost-obs-radius 5 --step-timeout 1 --start-mode stochastic
"""
import sys
from pathlib import Path
from collections import deque
import random
import numpy as np

# Add src folder
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

# UTILITIES

ALL_MOVES = [
    Move.UP,
    Move.DOWN,
    Move.LEFT,
    Move.RIGHT
]


def apply_move(pos, move):
    dr, dc = move.value
    return (pos[0] + dr, pos[1] + dc)


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def is_valid(pos, map_state):
    r, c = pos

    if r < 0 or r >= map_state.shape[0]:
        return False

    if c < 0 or c >= map_state.shape[1]:
        return False

    # Cannot walk through wall
    return map_state[r, c] == 0


def get_neighbors(pos, map_state):
    neighbors = []

    for move in ALL_MOVES:
        nxt = apply_move(pos, move)

        if is_valid(nxt, map_state):
            neighbors.append((nxt, move))

    return neighbors


def bfs(start, goal, map_state):
    """
    Standard BFS shortest path.
    Returns list of moves.
    """

    queue = deque()
    queue.append((start, []))

    visited = set()
    visited.add(start)

    while queue:

        current, path = queue.popleft()

        if current == goal:
            return path

        for nxt, move in get_neighbors(current, map_state):

            if nxt not in visited:
                visited.add(nxt)
                queue.append((nxt, path + [move]))

    return []


def bfs_furthest(start, threat_pos, map_state):
    """
    Find reachable position furthest from threat.
    """

    queue = deque([start])
    visited = {start}

    best_pos = start
    best_dist = manhattan(start, threat_pos)

    while queue:

        current = queue.popleft()

        dist = manhattan(current, threat_pos)

        if dist > best_dist:
            best_dist = dist
            best_pos = current

        for nxt, _ in get_neighbors(current, map_state):

            if nxt not in visited:
                visited.add(nxt)
                queue.append(nxt)

    return best_pos


# PACMAN 

class PacmanAgent(BasePacmanAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "Blind BFS Pacman"

        self.pacman_speed = int(kwargs.get("pacman_speed", 1))

        self.last_seen_enemy = None

        self.previous_positions = deque(maxlen=6)


    def step(
        self,
        map_state,
        my_position,
        enemy_position,
        step_number
    ):

        self.previous_positions.append(my_position)

        # SEE ENEMY (inside vison radius)

        if enemy_position is not None:

            self.last_seen_enemy = enemy_position

            path = bfs(
                my_position,
                enemy_position,
                map_state
            )

            if path:
                return self._path_to_action(
                    my_position,
                    enemy_position,
                    map_state
                )

        # LAST KNOWN ENEMY

        if self.last_seen_enemy is not None:

            path = bfs(
                my_position,
                self.last_seen_enemy,
                map_state
            )

            if path:
                return self._path_to_action(
                    my_position,
                    self.last_seen_enemy,
                    map_state
                )

        # EXPLORE UNKNOWN AREA

        move = self._explore_unknown(
            my_position,
            map_state
        )

        return move

    # Chose the best option for pacman, go 1 cell or 2 cell
    def _path_to_action(self, my_position, enemy_position, map_state):
        
        path = bfs(
            my_position,
            enemy_position,
            map_state
        )

        if not path:
            return (Move.STAY, 1)

        first_move = path[0]

        current = my_position
        valid_steps = 0

        # ALWAYS move maximum speed
        for move in path[:self.pacman_speed]:

            # stop if direction changes
            if move != first_move:
                break

            nxt = apply_move(current, move)

            if not is_valid(nxt, map_state):
                break

            current = nxt
            valid_steps += 1

        if valid_steps == 0:
            return (first_move, 1)

        return (first_move, valid_steps)

    # Solution 2: Move 2 cell if the second cell is safe, 1 cell if doesn't sure
    def _explore_unknown(self, my_position, map_state):
        
        # Solution 1: move only 1 cell (stuck in a chasing 5x5 loop)
        # best_move = None
        # best_score = -999999

        # for move in ALL_MOVES:

        #     nxt = apply_move(my_position, move)

        #     if not is_valid(nxt, map_state):
        #         continue

        #     score = 0

        #     # Prefer unseen nearby
        #     for nn, _ in get_neighbors(nxt, map_state):

        #         r, c = nn

        #         if map_state[r, c] == -1:
        #             score += 5

        #     # Avoid loops
        #     if nxt in self.previous_positions:
        #         score -= 3

        #     # Prefer mobility
        #     score += len(get_neighbors(nxt, map_state))

        #     if score > best_score:
        #         best_score = score
        #         best_move = move

        # if best_move is None:
        #     return (Move.STAY, 1)

        # return (best_move, 1)

        best_move = None
        best_score = -999999

        for move in ALL_MOVES:

            nxt = apply_move(my_position, move)

            if not is_valid(nxt, map_state):
                continue

            score = 0

            # Prefer unseen nearby
            for nn, _ in get_neighbors(nxt, map_state):

                r, c = nn

                if map_state[r, c] == -1:
                    score += 5

            # Avoid loops
            if nxt in self.previous_positions:
                score -= 3

            # Prefer mobility
            mobility = len(get_neighbors(nxt, map_state))
            score += mobility

            if score > best_score:
                best_score = score
                best_move = move

        if best_move is None:
            return (Move.STAY, 1)

        # SAFE 2-STEP LOGIC

        if self.pacman_speed >= 2:

            first = apply_move(my_position, best_move)
            second = apply_move(first, best_move)

            # only use 2-step if second step is safe
            if is_valid(second, map_state):

                second_mobility = len(
                    get_neighbors(second, map_state)
                )

                # avoid dead-end / corridor traps
                if second_mobility >= 2:

                    # avoid strong loops
                    if second not in self.previous_positions:

                        return (best_move, 2)

        return (best_move, 1)

        


# GHOST

class GhostAgent(BaseGhostAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.name = "Blind Escape Ghost"

        self.last_seen_enemy = None

        self.previous_positions = deque(maxlen=6)


    def step(
        self,
        map_state,
        my_position,
        enemy_position,
        step_number
    ):

        self.previous_positions.append(my_position)

        # SEE PACMAN

        if enemy_position is not None:

            self.last_seen_enemy = enemy_position

            target = bfs_furthest(
                my_position,
                enemy_position,
                map_state
            )

            path = bfs(
                my_position,
                target,
                map_state
            )

            if path:
                return path[0]

        # EXPLORE / EVADE

        return self._random_safe_move(
            my_position,
            map_state
        )


    def _random_safe_move(self, my_position, map_state):

        candidates = []

        for move in ALL_MOVES:

            nxt = apply_move(my_position, move)

            if not is_valid(nxt, map_state):
                continue

            score = 0

            # Avoid loops
            if nxt in self.previous_positions:
                score -= 5

            # Prefer open space
            score += len(get_neighbors(nxt, map_state))

            # Prefer unexplored
            for nn, _ in get_neighbors(nxt, map_state):

                r, c = nn

                if map_state[r, c] == -1:
                    score += 2

            candidates.append((score, move))

        if not candidates:
            return Move.STAY

        candidates.sort(reverse=True, key=lambda x: x[0])

        return candidates[0][1]