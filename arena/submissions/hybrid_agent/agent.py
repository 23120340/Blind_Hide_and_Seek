"""
Hybrid lookahead/safety agents for Blind Hide and Seek.

Pacman keeps the shallow lookahead hunter from lookahead_agent. Ghost blends
that worst-case simulation with stronger maze-safety heuristics from
codex_agent: avoid dead ends, prefer junctions, and value maze distance.
"""

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
MOVES_WITH_STAY = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]


class GridTools:
    def _setup(self, seed=73):
        self.known_map = None
        self.visited = {}
        self.last_enemy = None
        self.recent = deque(maxlen=10)
        self.rng = random.Random(seed)

    def _observe(self, map_state, my_position, enemy_position):
        if self.known_map is None or self.known_map.shape != map_state.shape:
            self.known_map = np.full(map_state.shape, -1, dtype=int)

        mask = map_state != -1
        self.known_map[mask] = map_state[mask]
        self.known_map[my_position] = 0

        if enemy_position is not None:
            self.last_enemy = enemy_position
            self.known_map[enemy_position] = 0

        self.visited[my_position] = self.visited.get(my_position, 0) + 1
        self.recent.append(my_position)

    def _inside(self, pos, grid):
        return 0 <= pos[0] < grid.shape[0] and 0 <= pos[1] < grid.shape[1]

    def _open(self, pos, grid):
        return self._inside(pos, grid) and grid[pos] != 1

    def _move(self, pos, move):
        dr, dc = move.value
        return pos[0] + dr, pos[1] + dc

    def _neighbors(self, pos, grid, include_stay=False):
        moves = MOVES_WITH_STAY if include_stay else MOVES
        result = []
        for move in moves:
            nxt = self._move(pos, move)
            if move == Move.STAY or self._open(nxt, grid):
                result.append((nxt, move))
        return result

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _maze_distance(self, start, goal, grid, max_depth=60):
        if start == goal:
            return 0
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            pos, dist = queue.popleft()
            if dist >= max_depth:
                continue
            for nxt, _ in self._neighbors(pos, grid):
                if nxt in seen:
                    continue
                if nxt == goal:
                    return dist + 1
                seen.add(nxt)
                queue.append((nxt, dist + 1))
        return max_depth + self._manhattan(start, goal)

    def _bfs_path(self, start, goal, grid, max_depth=80):
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
                new_path = path + [move]
                if nxt == goal:
                    return new_path
                seen.add(nxt)
                queue.append((nxt, new_path))
        return []

    def _unknown_count(self, pos, grid):
        count = 0
        for move in MOVES:
            nxt = self._move(pos, move)
            if self._inside(nxt, grid) and grid[nxt] == -1:
                count += 1
        return count

    def _frontier_target(self, start):
        queue = deque([(start, 0)])
        seen = {start}
        best = None
        best_score = -10**9
        while queue:
            pos, dist = queue.popleft()
            if dist > 30:
                continue
            visits = self.visited.get(pos, 0)
            score = self._unknown_count(pos, self.known_map) * 10 - dist - visits * 3
            if score > best_score and pos != start:
                best = pos
                best_score = score
            for nxt, _ in self._neighbors(pos, self.known_map):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, dist + 1))
        return best


class PacmanAgent(BasePacmanAgent, GridTools):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Hybrid Lookahead Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self._setup(seed=101)

    def step(self, map_state, my_position, enemy_position, step_number):
        self._observe(map_state, my_position, enemy_position)

        if enemy_position is not None:
            return self._best_lookahead_action(my_position, enemy_position, map_state)

        target = self.last_enemy if self.last_enemy and self.last_enemy != my_position else self._frontier_target(my_position)
        if target is not None:
            path = self._bfs_path(my_position, target, self.known_map)
            if path:
                return self._path_action(my_position, path, map_state)

        return self._fallback(my_position, map_state)

    def _candidate_actions(self, pos, grid):
        actions = []
        for move in MOVES:
            current = pos
            for steps in range(1, self.pacman_speed + 1):
                nxt = self._move(current, move)
                if not self._open(nxt, grid):
                    break
                actions.append((move, steps, nxt))
                current = nxt
        actions.append((Move.STAY, 1, pos))
        return actions

    def _best_lookahead_action(self, my_pos, enemy_pos, grid):
        ghost_moves = self._neighbors(enemy_pos, self.known_map, include_stay=True)
        best = (Move.STAY, 1)
        best_score = -10**9

        for move, steps, pac_next in self._candidate_actions(my_pos, grid):
            worst_score = 10**9
            for ghost_next, _ in ghost_moves:
                dist = self._manhattan(pac_next, ghost_next)
                if dist < 2:
                    score = 10000 - steps
                else:
                    maze = self._maze_distance(pac_next, ghost_next, self.known_map, max_depth=50)
                    score = -maze * 8 - dist * 2 - steps * 0.2
                worst_score = min(worst_score, score)

            if worst_score > best_score:
                best_score = worst_score
                best = (move, steps)

        return best

    def _path_action(self, pos, path, grid):
        first = path[0]
        current = pos
        steps = 0
        for move in path[: self.pacman_speed]:
            if move != first:
                break
            nxt = self._move(current, move)
            if not self._open(nxt, grid):
                break
            current = nxt
            steps += 1
        return (first, max(1, steps)) if steps else self._fallback(pos, grid)

    def _fallback(self, pos, grid):
        candidates = []
        for nxt, move in self._neighbors(pos, grid):
            score = self._unknown_count(nxt, self.known_map) * 5 - self.visited.get(nxt, 0)
            candidates.append((score, move))
        if not candidates:
            return Move.STAY, 1
        candidates.sort(reverse=True, key=lambda item: item[0])
        return candidates[0][1], 1


class GhostAgent(BaseGhostAgent, GridTools):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Hybrid Safety Ghost"
        self._setup(seed=202)

    def step(self, map_state, my_position, enemy_position, step_number):
        self._observe(map_state, my_position, enemy_position)
        threat = enemy_position or self.last_enemy

        if threat is not None:
            return self._best_escape_move(my_position, threat, map_state)

        candidates = []
        for nxt, move in self._neighbors(my_position, map_state):
            score = self._safe_cell_score(nxt, None)
            candidates.append((score, move))
        if not candidates:
            return Move.STAY
        candidates.sort(reverse=True, key=lambda item: item[0])
        return candidates[0][1]

    def _pacman_next_positions(self, pac_pos):
        positions = [pac_pos]
        for move in MOVES:
            current = pac_pos
            for _ in range(2):
                nxt = self._move(current, move)
                if not self._open(nxt, self.known_map):
                    break
                positions.append(nxt)
                current = nxt
        return positions

    def _best_escape_move(self, my_pos, threat, grid):
        pacman_positions = self._pacman_next_positions(threat)
        best_move = Move.STAY
        best_score = -10**9

        for ghost_next, move in self._neighbors(my_pos, grid, include_stay=True):
            worst_distance = min(self._manhattan(ghost_next, pac_next) for pac_next in pacman_positions)
            worst_maze = min(self._maze_distance(ghost_next, pac_next, self.known_map, max_depth=28) for pac_next in pacman_positions)
            score = self._safe_cell_score(ghost_next, threat)
            score += worst_distance * 10
            score += min(worst_maze, 28) * 5
            if worst_distance < 2:
                score -= 500
            elif worst_distance < 4:
                score -= 60
            if move == Move.STAY:
                score -= 12
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _safe_cell_score(self, pos, threat):
        exits = len(self._neighbors(pos, self.known_map))
        score = exits * 10 + self._unknown_count(pos, self.known_map) * 2
        score -= self.visited.get(pos, 0) * 2.5
        if pos in list(self.recent)[-4:]:
            score -= 7
        if exits <= 1:
            score -= 70
        elif exits == 2 and self._near_dead_end(pos):
            score -= 25
        if threat is not None:
            score += min(28, self._maze_distance(pos, threat, self.known_map, max_depth=28)) * 5
        return score

    def _near_dead_end(self, pos):
        queue = deque([(pos, 0)])
        seen = {pos}
        junction_seen = False
        while queue:
            current, dist = queue.popleft()
            if dist > 5:
                continue
            exits = len(self._neighbors(current, self.known_map))
            if exits >= 3:
                junction_seen = True
                break
            for nxt, _ in self._neighbors(current, self.known_map):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, dist + 1))
        return not junction_seen
