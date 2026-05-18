"""Trap-aware agent: chase routes that reduce enemy escape options."""

import heapq
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


class Grid:
    def _init_grid(self):
        self.known = None
        self.visits = {}
        self.last_enemy = None
        self.recent = deque(maxlen=10)

    def _observe(self, grid, me, enemy):
        if self.known is None or self.known.shape != grid.shape:
            self.known = np.full(grid.shape, -1, dtype=int)
        visible = grid != -1
        self.known[visible] = grid[visible]
        self.known[me] = 0
        if enemy is not None:
            self.last_enemy = enemy
            self.known[enemy] = 0
        self.visits[me] = self.visits.get(me, 0) + 1
        self.recent.append(me)

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

    def _astar(self, start, goal, grid, limit=700):
        if start == goal:
            return []
        heap = [(self._manhattan(start, goal), 0, start)]
        best = {start: 0}
        parent = {}
        expansions = 0
        while heap and expansions < limit:
            _, cost, pos = heapq.heappop(heap)
            expansions += 1
            if pos == goal:
                path = []
                while pos in parent:
                    pos, move = parent[pos]
                    path.append(move)
                path.reverse()
                return path
            for nxt, move in self._neighbors(pos, grid):
                penalty = 2 if grid[nxt] == -1 else 0
                new_cost = cost + 1 + penalty
                if new_cost >= best.get(nxt, 10**9):
                    continue
                best[nxt] = new_cost
                parent[nxt] = (pos, move)
                heapq.heappush(heap, (new_cost + self._manhattan(nxt, goal), new_cost, nxt))
        return []

    def _maze_distance(self, start, goal, grid, cap=50):
        if start == goal:
            return 0
        q = deque([(start, 0)])
        seen = {start}
        while q:
            pos, dist = q.popleft()
            if dist >= cap:
                continue
            for nxt, _ in self._neighbors(pos, grid):
                if nxt in seen:
                    continue
                if nxt == goal:
                    return dist + 1
                seen.add(nxt)
                q.append((nxt, dist + 1))
        return cap + self._manhattan(start, goal)

    def _unknown_adjacent(self, pos):
        total = 0
        for move in MOVES:
            nxt = self._move(pos, move)
            if self._inside(nxt, self.known) and self.known[nxt] == -1:
                total += 1
        return total

    def _escape_options(self, pos):
        return len(self._neighbors(pos, self.known))


class PacmanAgent(BasePacmanAgent, Grid):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Trap Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self._init_grid()

    def step(self, map_state, my_position, enemy_position, step_number):
        self._observe(map_state, my_position, enemy_position)

        if enemy_position is not None:
            target = self._trap_target(enemy_position)
        else:
            target = self.last_enemy or self._info_target(my_position)

        if target is not None:
            path = self._astar(my_position, target, self.known)
            if path:
                return self._path_action(my_position, path, map_state)
        return self._fallback(my_position, map_state)

    def _trap_target(self, enemy):
        options = [enemy] + [nxt for nxt, _ in self._neighbors(enemy, self.known)]
        best = enemy
        best_score = -10**9
        for pos in options:
            escape = self._escape_options(pos)
            unknown = self._unknown_adjacent(pos)
            score = -escape * 12 - unknown * 4 - self._manhattan(pos, enemy)
            if score > best_score:
                best = pos
                best_score = score
        return best

    def _info_target(self, start):
        q = deque([(start, 0)])
        seen = {start}
        best = None
        best_score = -10**9
        while q:
            pos, dist = q.popleft()
            if dist > 30:
                continue
            score = self._unknown_adjacent(pos) * 8 - dist - self.visits.get(pos, 0) * 3
            if score > best_score and pos != start:
                best = pos
                best_score = score
            for nxt, _ in self._neighbors(pos, self.known):
                if nxt not in seen:
                    seen.add(nxt)
                    q.append((nxt, dist + 1))
        return best

    def _path_action(self, pos, path, grid):
        first = path[0]
        steps = 0
        cur = pos
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
        best = (Move.STAY, -10**9)
        for nxt, move in self._neighbors(pos, grid):
            score = self._unknown_adjacent(nxt) * 4 - self.visits.get(nxt, 0)
            if score > best[1]:
                best = (move, score)
        return best[0], 1


class GhostAgent(BaseGhostAgent, Grid):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Trap-Aware Ghost"
        self._init_grid()

    def step(self, map_state, my_position, enemy_position, step_number):
        self._observe(map_state, my_position, enemy_position)
        threat = enemy_position or self.last_enemy

        best_move = Move.STAY
        best_score = -10**9
        for nxt, move in self._neighbors(my_position, map_state, stay=True):
            score = self._score(nxt, move, threat)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _score(self, pos, move, threat):
        exits = self._escape_options(pos)
        score = exits * 16 + self._unknown_adjacent(pos) * 2
        score -= self.visits.get(pos, 0) * 2.5
        if exits <= 1:
            score -= 90
        if pos in list(self.recent)[-4:]:
            score -= 9
        if move == Move.STAY:
            score -= 12
        if threat is not None:
            score += min(35, self._maze_distance(pos, threat, self.known, cap=35)) * 5
            direct = self._manhattan(pos, threat)
            if direct < 2:
                score -= 600
            elif direct < 4:
                score -= 80
        return score
