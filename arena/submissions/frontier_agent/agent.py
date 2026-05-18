"""Frontier-search seeker and farthest-safe-cell hider."""

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


class Tools:
    def _init(self):
        self.known = None
        self.visits = {}
        self.last_enemy = None
        self.recent = deque(maxlen=8)

    def _observe(self, grid, pos, enemy):
        if self.known is None or self.known.shape != grid.shape:
            self.known = np.full(grid.shape, -1, dtype=int)
        mask = grid != -1
        self.known[mask] = grid[mask]
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

    def _step(self, pos, move):
        dr, dc = move.value
        return pos[0] + dr, pos[1] + dc

    def _neighbors(self, pos, grid, stay=False):
        moves = MOVES_STAY if stay else MOVES
        out = []
        for move in moves:
            nxt = self._step(pos, move)
            if move == Move.STAY or self._open(nxt, grid):
                out.append((nxt, move))
        return out

    def _dist(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _bfs_path(self, start, goal, grid, limit=120):
        if start == goal:
            return []
        q = deque([(start, [])])
        seen = {start}
        while q:
            pos, path = q.popleft()
            if len(path) >= limit:
                continue
            for nxt, move in self._neighbors(pos, grid):
                if nxt in seen:
                    continue
                new_path = path + [move]
                if nxt == goal:
                    return new_path
                seen.add(nxt)
                q.append((nxt, new_path))
        return []

    def _distance_map(self, start, grid, max_depth=80):
        q = deque([start])
        dist = {start: 0}
        while q:
            pos = q.popleft()
            if dist[pos] >= max_depth:
                continue
            for nxt, _ in self._neighbors(pos, grid):
                if nxt not in dist:
                    dist[nxt] = dist[pos] + 1
                    q.append(nxt)
        return dist

    def _unknown_adjacent(self, pos):
        total = 0
        for move in MOVES:
            nxt = self._step(pos, move)
            if self._inside(nxt, self.known) and self.known[nxt] == -1:
                total += 1
        return total


class PacmanAgent(BasePacmanAgent, Tools):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Frontier Pacman"
        self._init()

    def step(self, map_state, my_position, enemy_position, step_number):
        self._observe(map_state, my_position, enemy_position)

        target = enemy_position or self._frontier_target(my_position)
        if target is None and self.last_enemy is not None:
            target = self.last_enemy

        if target is not None:
            path = self._bfs_path(my_position, target, self.known)
            if path:
                return self._path_action(my_position, path, map_state)

        return self._best_visible_move(my_position, map_state)

    def _frontier_target(self, start):
        distances = self._distance_map(start, self.known, max_depth=45)
        best = None
        best_score = -10**9
        for pos, dist in distances.items():
            if pos == start:
                continue
            score = self._unknown_adjacent(pos) * 12 - dist * 0.8 - self.visits.get(pos, 0) * 4
            if self.last_enemy is not None:
                score -= self._dist(pos, self.last_enemy) * 0.15
            if score > best_score:
                best = pos
                best_score = score
        return best

    def _path_action(self, pos, path, grid):
        first = path[0]
        cur = pos
        steps = 0
        for move in path[: self.pacman_speed]:
            if move != first:
                break
            nxt = self._step(cur, move)
            if not self._open(nxt, grid):
                break
            cur = nxt
            steps += 1
        return first, max(1, steps)

    def _best_visible_move(self, pos, grid):
        choices = []
        for nxt, move in self._neighbors(pos, grid):
            choices.append((self._unknown_adjacent(nxt) * 5 - self.visits.get(nxt, 0), move))
        if not choices:
            return Move.STAY, 1
        choices.sort(reverse=True, key=lambda item: item[0])
        return choices[0][1], 1


class GhostAgent(BaseGhostAgent, Tools):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Farthest Frontier Ghost"
        self._init()

    def step(self, map_state, my_position, enemy_position, step_number):
        self._observe(map_state, my_position, enemy_position)
        threat = enemy_position or self.last_enemy

        if threat is None:
            return self._safe_wander(my_position, map_state)

        target = self._farthest_safe_target(my_position, threat)
        path = self._bfs_path(my_position, target, self.known, limit=40) if target else []
        if path:
            return path[0]
        return self._safe_wander(my_position, map_state, threat)

    def _farthest_safe_target(self, start, threat):
        distances = self._distance_map(start, self.known, max_depth=35)
        best = start
        best_score = -10**9
        for pos, dist in distances.items():
            exits = len(self._neighbors(pos, self.known))
            score = self._dist(pos, threat) * 5 + dist * 0.5 + exits * 4
            score -= self.visits.get(pos, 0) * 3
            if exits <= 1:
                score -= 50
            if pos in list(self.recent)[-4:]:
                score -= 10
            if score > best_score:
                best = pos
                best_score = score
        return best

    def _safe_wander(self, pos, grid, threat=None):
        choices = []
        for nxt, move in self._neighbors(pos, grid):
            exits = len(self._neighbors(nxt, self.known))
            score = exits * 6 - self.visits.get(nxt, 0) * 2
            if threat is not None:
                score += self._dist(nxt, threat) * 4
            if nxt in list(self.recent)[-3:]:
                score -= 6
            choices.append((score, move))
        if not choices:
            return Move.STAY
        choices.sort(reverse=True, key=lambda item: item[0])
        return choices[0][1]
