"""Small Monte Carlo rollout agent for visible opponent situations."""

import random
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


class RolloutTools:
    def _init_rollout(self, seed):
        self.known = None
        self.visits = {}
        self.last_enemy = None
        self.rng = random.Random(seed)

    def _observe(self, grid, pos, enemy):
        if self.known is None or self.known.shape != grid.shape:
            self.known = np.full(grid.shape, -1, dtype=int)
        seen = grid != -1
        self.known[seen] = grid[seen]
        self.known[pos] = 0
        if enemy is not None:
            self.last_enemy = enemy
            self.known[enemy] = 0
        self.visits[pos] = self.visits.get(pos, 0) + 1

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

    def _bfs_path(self, start, goal, grid, max_depth=90):
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

    def _random_next(self, pos, grid, stay=False):
        neighbors = self._neighbors(pos, grid, stay=stay)
        if not neighbors:
            return pos
        return self.rng.choice(neighbors)[0]

    def _unknown_adjacent(self, pos):
        total = 0
        for move in MOVES:
            nxt = self._move(pos, move)
            if self._inside(nxt, self.known) and self.known[nxt] == -1:
                total += 1
        return total

    def _frontier(self, start):
        queue = deque([(start, 0)])
        seen = {start}
        best = None
        best_score = -10**9
        while queue:
            pos, dist = queue.popleft()
            if dist > 28:
                continue
            score = self._unknown_adjacent(pos) * 8 - dist - self.visits.get(pos, 0) * 2
            if score > best_score and pos != start:
                best = pos
                best_score = score
            for nxt, _ in self._neighbors(pos, self.known):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, dist + 1))
        return best


class PacmanAgent(BasePacmanAgent, RolloutTools):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Monte Carlo Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self._init_rollout(seed=303)

    def step(self, map_state, my_position, enemy_position, step_number):
        self._observe(map_state, my_position, enemy_position)

        if enemy_position is not None:
            return self._rollout_action(my_position, enemy_position, map_state)

        target = self.last_enemy if self.last_enemy and self.last_enemy != my_position else self._frontier(my_position)
        if target is not None:
            path = self._bfs_path(my_position, target, self.known)
            if path:
                return self._path_action(my_position, path, map_state)
        return Move.STAY, 1

    def _actions(self, pos, grid):
        actions = []
        for move in MOVES:
            cur = pos
            for steps in range(1, self.pacman_speed + 1):
                nxt = self._move(cur, move)
                if not self._open(nxt, grid):
                    break
                actions.append((move, steps, nxt))
                cur = nxt
        return actions or [(Move.STAY, 1, pos)]

    def _rollout_action(self, me, enemy, grid):
        best = (Move.STAY, 1)
        best_score = -10**9
        for move, steps, pac_next in self._actions(me, grid):
            total = 0
            for _ in range(18):
                p = pac_next
                g = enemy
                score = 0
                for depth in range(4):
                    if self._manhattan(p, g) < 2:
                        score += 1000 - depth * 30
                        break
                    g = self._random_next(g, self.known, stay=True)
                    path = self._bfs_path(p, g, self.known, max_depth=12)
                    if path:
                        p = self._advance(p, path[0], self.pacman_speed, self.known)
                    score -= self._manhattan(p, g) * 5
                total += score
            avg = total / 18
            if avg > best_score:
                best_score = avg
                best = (move, steps)
        return best

    def _advance(self, pos, move, steps, grid):
        cur = pos
        for _ in range(steps):
            nxt = self._move(cur, move)
            if not self._open(nxt, grid):
                break
            cur = nxt
        return cur

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


class GhostAgent(BaseGhostAgent, RolloutTools):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Monte Carlo Ghost"
        self._init_rollout(seed=404)

    def step(self, map_state, my_position, enemy_position, step_number):
        self._observe(map_state, my_position, enemy_position)
        threat = enemy_position or self.last_enemy
        if threat is not None:
            return self._rollout_escape(my_position, threat, map_state)
        return self._wander(my_position, map_state)

    def _rollout_escape(self, me, threat, grid):
        best_move = Move.STAY
        best_score = -10**9
        for ghost_next, move in self._neighbors(me, grid, stay=True):
            total = 0
            for _ in range(16):
                g = ghost_next
                p = threat
                score = 0
                for depth in range(4):
                    if self._manhattan(p, g) < 2:
                        score -= 1000 - depth * 20
                        break
                    p = self._random_next(p, self.known)
                    g = self._random_next(g, self.known, stay=True)
                    score += self._manhattan(p, g) * 6 + len(self._neighbors(g, self.known)) * 3
                total += score
            avg = total / 16 - self.visits.get(ghost_next, 0) * 2
            if avg > best_score:
                best_score = avg
                best_move = move
        return best_move

    def _wander(self, pos, grid):
        choices = []
        for nxt, move in self._neighbors(pos, grid):
            score = self._unknown_adjacent(nxt) * 3 + len(self._neighbors(nxt, self.known)) * 5
            score -= self.visits.get(nxt, 0) * 2
            choices.append((score, move))
        if not choices:
            return Move.STAY
        choices.sort(reverse=True, key=lambda item: item[0])
        return choices[0][1]
