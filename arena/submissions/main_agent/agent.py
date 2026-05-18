"""
Main tournament agent.

Pacman uses fast frontier/BFS pursuit. Ghost uses a blended safety evaluator:
direct Pacman reachability, maze distance, local escape options, and short
horizon escape potential.
"""

import sys
from collections import deque
from pathlib import Path

import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import GhostAgent as BaseGhostAgent
from agent_interface import PacmanAgent as BasePacmanAgent
from environment import Move


MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
MOVES_STAY = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
DEFAULT_PACMAN_START = (15, 10)
DEFAULT_GHOST_START = (9, 10)
DEFAULT_LAYOUT = [
    "#####################",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...................#",
    "#.###.#.#####.#.###.#",
    "#.....#...#...#.....#",
    "#####.###.#.###.#####",
    "#...#.#.......#.#...#",
    "#####.#.#####.#.#####",
    "#.........G.........#",
    "#####.#.#####.#.#####",
    "#...#.#.......#.#...#",
    "#####.#.#####.#.#####",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...#.....P.....#...#",
    "###.#.#.#####.#.#.###",
    "#.....#...#...#.....#",
    "#.#######.#.#######.#",
    "#...................#",
    "#####################",
]


def _default_map():
    grid = np.zeros((len(DEFAULT_LAYOUT), len(DEFAULT_LAYOUT[0])), dtype=int)
    for row_idx, row in enumerate(DEFAULT_LAYOUT):
        for col_idx, cell in enumerate(row):
            grid[row_idx, col_idx] = 1 if cell == "#" else 0
    return grid


DEFAULT_MAP = _default_map()


class ArenaTools:
    def _init_memory(self):
        self.known = None
        self.visits = {}
        self.last_enemy = None
        self.recent = deque(maxlen=10)
        self.fixed_map_mode = False
        self.last_prediction_step = 0

    def _maybe_enable_fixed_map(self, grid, pos, assumed_enemy):
        if self.fixed_map_mode:
            return
        if grid.shape != DEFAULT_MAP.shape:
            return
        walls_match = np.array_equal(grid == 1, DEFAULT_MAP == 1)
        if not walls_match:
            return
        if pos not in (DEFAULT_PACMAN_START, DEFAULT_GHOST_START):
            return
        self.fixed_map_mode = True
        self.known = DEFAULT_MAP.copy()
        self.last_enemy = assumed_enemy

    def _observe(self, grid, pos, enemy, assumed_enemy=None):
        if assumed_enemy is not None:
            self._maybe_enable_fixed_map(grid, pos, assumed_enemy)

        if self.known is None or self.known.shape != grid.shape:
            self.known = np.full(grid.shape, -1, dtype=int)

        visible = grid != -1
        self.known[visible] = grid[visible]
        self.known[pos] = 0

        if enemy is not None:
            self.last_enemy = enemy
            self.known[enemy] = 0

        self.visits[pos] = self.visits.get(pos, 0) + 1
        self.recent.append(pos)

    def _inside(self, pos, grid):
        return 0 <= pos[0] < grid.shape[0] and 0 <= pos[1] < grid.shape[1]

    def _open(self, pos, grid):
        return self._inside(pos, grid) and grid[pos] != 1

    def _move(self, pos, move):
        return pos[0] + move.value[0], pos[1] + move.value[1]

    def _neighbors(self, pos, grid, stay=False):
        out = []
        for move in (MOVES_STAY if stay else MOVES):
            nxt = self._move(pos, move)
            if move == Move.STAY or self._open(nxt, grid):
                out.append((nxt, move))
        return out

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _bfs_path(self, start, goal, grid, max_depth=120):
        if start == goal:
            return []

        queue = deque([(start, [])])
        seen = {start}
        while queue:
            pos, path = queue.popleft()
            if len(path) >= max_depth:
                continue
            for nxt, move in self._neighbors(pos, grid):
                if nxt in seen:
                    continue
                next_path = path + [move]
                if nxt == goal:
                    return next_path
                seen.add(nxt)
                queue.append((nxt, next_path))
        return []

    def _distance_map(self, start, grid, max_depth=50):
        queue = deque([start])
        dist = {start: 0}
        while queue:
            pos = queue.popleft()
            if dist[pos] >= max_depth:
                continue
            for nxt, _ in self._neighbors(pos, grid):
                if nxt not in dist:
                    dist[nxt] = dist[pos] + 1
                    queue.append(nxt)
        return dist

    def _maze_distance(self, start, goal, grid, max_depth=50):
        if start == goal:
            return 0
        distances = self._distance_map(start, grid, max_depth=max_depth)
        if goal in distances:
            return distances[goal]
        return max_depth + self._manhattan(start, goal)

    def _unknown_adjacent(self, pos):
        total = 0
        for move in MOVES:
            nxt = self._move(pos, move)
            if self._inside(nxt, self.known) and self.known[nxt] == -1:
                total += 1
        return total

    def _advance_toward(self, start, target, speed):
        path = self._bfs_path(start, target, self.known, max_depth=80)
        cur = start
        for move in path[:speed]:
            nxt = self._move(cur, move)
            if not self._open(nxt, self.known):
                break
            cur = nxt
        return cur


class PacmanAgent(BasePacmanAgent, ArenaTools):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Main Frontier Hunter"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self._init_memory()

    def step(self, map_state, my_position, enemy_position, step_number):
        self._observe(map_state, my_position, enemy_position, DEFAULT_GHOST_START)

        if enemy_position is not None:
            path = self._bfs_path(my_position, enemy_position, self.known)
            if path:
                return self._path_action(my_position, path, map_state)

        target = None
        if self.fixed_map_mode and self.last_enemy is not None and self.last_enemy != my_position:
            target = self.last_enemy
        if target is None:
            target = self._frontier_target(my_position)
        if target is None and self.last_enemy is not None and self.last_enemy != my_position:
            target = self.last_enemy

        if target is not None:
            path = self._bfs_path(my_position, target, self.known)
            if path:
                return self._path_action(my_position, path, map_state)

        return self._fallback(my_position, map_state)

    def _frontier_target(self, start):
        distances = self._distance_map(start, self.known, max_depth=45)
        best = None
        best_score = -10**9
        for pos, dist in distances.items():
            if pos == start:
                continue
            score = self._unknown_adjacent(pos) * 12
            score -= dist * 0.8
            score -= self.visits.get(pos, 0) * 4
            if self.last_enemy is not None:
                score -= self._manhattan(pos, self.last_enemy) * 0.12
            if score > best_score:
                best_score = score
                best = pos
        return best

    def _path_action(self, pos, path, grid):
        first = path[0]
        cur = pos
        steps = 0
        for move in path[: self.pacman_speed]:
            if move != first:
                break
            nxt = self._move(cur, move)
            if not self._open(nxt, grid):
                break
            cur = nxt
            steps += 1
        return first, max(1, steps)

    def _fallback(self, pos, grid):
        best_move = Move.STAY
        best_score = -10**9
        for nxt, move in self._neighbors(pos, grid):
            score = self._unknown_adjacent(nxt) * 5 - self.visits.get(nxt, 0) * 2
            if score > best_score:
                best_score = score
                best_move = move
        return best_move, 1


class GhostAgent(BaseGhostAgent, ArenaTools):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Main Escape Hider"
        self._init_memory()
        self.fixed_opening = [
            Move.RIGHT,
            Move.RIGHT,
            Move.RIGHT,
            Move.UP,
            Move.UP,
            Move.LEFT,
        ]

    def step(self, map_state, my_position, enemy_position, step_number):
        self._observe(map_state, my_position, enemy_position, DEFAULT_PACMAN_START)
        threat = enemy_position or self.last_enemy

        if self.fixed_map_mode and step_number <= len(self.fixed_opening):
            opening_move = self.fixed_opening[step_number - 1]
            opening_pos = self._move(my_position, opening_move)
            if self._open(opening_pos, map_state):
                return opening_move

        best_move = Move.STAY
        best_score = -10**9
        for nxt, move in self._neighbors(my_position, map_state, stay=True):
            score = self._escape_score(nxt, move, threat)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _escape_score(self, pos, move, threat):
        exits = len(self._neighbors(pos, self.known))
        potential = self._escape_potential(pos, threat)

        score = exits * 10
        score += potential * 2.5
        score += self._unknown_adjacent(pos) * 1.5
        score -= self.visits.get(pos, 0) * 2.5

        if pos in list(self.recent)[-4:]:
            score -= 8
        if exits <= 1:
            score -= 85
        elif exits == 2 and self._corridor_without_junction(pos):
            score -= 22
        if move == Move.STAY:
            score -= 12

        if threat is not None:
            pacman_reach = self._pacman_reachable_positions(threat)
            min_direct = min(self._manhattan(pos, pac_pos) for pac_pos in pacman_reach)
            min_maze = min(self._maze_distance(pos, pac_pos, self.known, max_depth=32) for pac_pos in pacman_reach)

            score += min(self._maze_distance(pos, threat, self.known, max_depth=35), 35) * 4.5
            score += min(min_maze, 32) * 5
            score += min_direct * 8

            if min_direct < 2:
                score -= 700
            elif min_direct < 4:
                score -= 90

        return score

    def _pacman_reachable_positions(self, pacman_pos):
        positions = [pacman_pos]
        for move in MOVES:
            cur = pacman_pos
            for _ in range(2):
                nxt = self._move(cur, move)
                if not self._open(nxt, self.known):
                    break
                positions.append(nxt)
                cur = nxt
        return positions

    def _escape_potential(self, start, threat):
        distances = self._distance_map(start, self.known, max_depth=7)
        best = 0
        for pos, dist in distances.items():
            exits = len(self._neighbors(pos, self.known))
            score = exits * 3 - dist * 0.5 - self.visits.get(pos, 0)
            if threat is not None:
                score += min(self._maze_distance(pos, threat, self.known, max_depth=25), 25)
            best = max(best, score)
        return best

    def _corridor_without_junction(self, start):
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            pos, dist = queue.popleft()
            if dist > 5:
                continue
            if len(self._neighbors(pos, self.known)) >= 3:
                return False
            for nxt, _ in self._neighbors(pos, self.known):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, dist + 1))
        return True
