"""
Memory-based agents for the Blind Hide and Seek arena.

The implementation is intentionally self-contained so the submission folder can
be zipped directly. It combines classic search with lightweight opponent-aware
heuristics that stay comfortably below the per-step time limit.
"""

import heapq
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


CARDINAL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]
ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]


class SearchMemoryMixin:
    def _init_memory(self):
        self.known_map = None
        self.visited_count = {}
        self.last_known_enemy_pos = None
        self.previous_positions = deque(maxlen=8)
        self.rng = random.Random(2026)

    def _update_memory(self, map_state, my_position, enemy_position):
        if self.known_map is None or self.known_map.shape != map_state.shape:
            self.known_map = np.full(map_state.shape, -1, dtype=int)

        visible_or_wall = map_state != -1
        self.known_map[visible_or_wall] = map_state[visible_or_wall]
        self.known_map[my_position] = 0

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.known_map[enemy_position] = 0

        self.visited_count[my_position] = self.visited_count.get(my_position, 0) + 1
        self.previous_positions.append(my_position)

    def _in_bounds(self, pos, grid):
        row, col = pos
        return 0 <= row < grid.shape[0] and 0 <= col < grid.shape[1]

    def _is_open(self, pos, grid):
        return self._in_bounds(pos, grid) and grid[pos] != 1

    def _is_known_open(self, pos, grid):
        return self._in_bounds(pos, grid) and grid[pos] == 0

    def _apply_move(self, pos, move):
        dr, dc = move.value
        return pos[0] + dr, pos[1] + dc

    def _valid_moves(self, pos, grid, include_stay=False, known_only=False):
        moves = ALL_MOVES if include_stay else CARDINAL_MOVES
        result = []
        for move in moves:
            nxt = self._apply_move(pos, move)
            if move == Move.STAY:
                result.append((nxt, move))
            elif known_only and self._is_known_open(nxt, grid):
                result.append((nxt, move))
            elif not known_only and self._is_open(nxt, grid):
                result.append((nxt, move))
        return result

    def _manhattan(self, a, b):
        return abs(a[0] - b[0]) + abs(a[1] - b[1])

    def _path_to_moves(self, came_from, goal):
        moves = []
        node = goal
        while node in came_from:
            node, move = came_from[node]
            moves.append(move)
        moves.reverse()
        return moves

    def _bfs_path(self, start, goal, grid, known_only=False, max_expansions=600):
        if start == goal:
            return []

        queue = deque([start])
        visited = {start}
        came_from = {}
        expansions = 0

        while queue and expansions < max_expansions:
            current = queue.popleft()
            expansions += 1
            for nxt, move in self._valid_moves(current, grid, known_only=known_only):
                if nxt in visited:
                    continue
                visited.add(nxt)
                came_from[nxt] = (current, move)
                if nxt == goal:
                    return self._path_to_moves(came_from, nxt)
                queue.append(nxt)
        return []

    def _astar_path(self, start, goal, grid, known_only=False, max_expansions=600):
        if start == goal:
            return []

        heap = [(self._manhattan(start, goal), 0, start)]
        best_cost = {start: 0}
        came_from = {}
        expansions = 0

        while heap and expansions < max_expansions:
            _, cost, current = heapq.heappop(heap)
            expansions += 1
            if current == goal:
                return self._path_to_moves(came_from, current)

            for nxt, move in self._valid_moves(current, grid, known_only=known_only):
                terrain_penalty = 2 if grid[nxt] == -1 else 0
                new_cost = cost + 1 + terrain_penalty
                if new_cost >= best_cost.get(nxt, 10**9):
                    continue
                best_cost[nxt] = new_cost
                came_from[nxt] = (current, move)
                priority = new_cost + self._manhattan(nxt, goal)
                heapq.heappush(heap, (priority, new_cost, nxt))
        return []

    def _distance_map(self, start, grid, max_depth=20):
        distances = {start: 0}
        queue = deque([start])
        while queue:
            current = queue.popleft()
            if distances[current] >= max_depth:
                continue
            for nxt, _ in self._valid_moves(current, grid):
                if nxt not in distances:
                    distances[nxt] = distances[current] + 1
                    queue.append(nxt)
        return distances

    def _unknown_neighbors(self, pos, grid):
        count = 0
        for move in CARDINAL_MOVES:
            nxt = self._apply_move(pos, move)
            if self._in_bounds(nxt, grid) and grid[nxt] == -1:
                count += 1
        return count


class PacmanAgent(BasePacmanAgent, SearchMemoryMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Codex Memory Hunter"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self._init_memory()

    def step(self, map_state, my_position, enemy_position, step_number):
        self._update_memory(map_state, my_position, enemy_position)

        if enemy_position is not None:
            path = self._hunt_path(my_position, enemy_position)
            if path:
                return self._action_from_path(my_position, path, map_state)

        target = self._choose_search_target(my_position, step_number)
        if target is not None:
            path = self._astar_path(my_position, target, self.known_map)
            if path:
                return self._action_from_path(my_position, path, map_state)

        return self._fallback_action(my_position, map_state)

    def _hunt_path(self, my_position, enemy_position):
        goals = [enemy_position]
        for nxt, _ in self._valid_moves(enemy_position, self.known_map):
            goals.append(nxt)

        best_path = []
        best_len = 10**9
        for goal in goals:
            path = self._astar_path(my_position, goal, self.known_map)
            if path and len(path) < best_len:
                best_path = path
                best_len = len(path)
        return best_path

    def _choose_search_target(self, my_position, step_number):
        if self.last_known_enemy_pos is not None and my_position != self.last_known_enemy_pos:
            return self.last_known_enemy_pos

        distances = self._distance_map(my_position, self.known_map, max_depth=28)
        best_target = None
        best_score = -10**9

        for pos, dist in distances.items():
            if pos == my_position:
                continue
            visits = self.visited_count.get(pos, 0)
            info_gain = self._unknown_neighbors(pos, self.known_map)
            center_bias = -0.05 * self._manhattan(pos, (self.known_map.shape[0] // 2, self.known_map.shape[1] // 2))
            score = info_gain * 8 - dist * 0.6 - visits * 3 + center_bias

            if step_number % 17 == 0:
                score += self.rng.random() * 0.5

            if score > best_score:
                best_score = score
                best_target = pos

        return best_target

    def _action_from_path(self, my_position, path, map_state):
        first = path[0]
        steps = 0
        current = my_position
        for move in path[: self.pacman_speed]:
            if move != first:
                break
            nxt = self._apply_move(current, move)
            if not self._is_open(nxt, map_state):
                break
            steps += 1
            current = nxt

        if steps <= 0:
            return self._fallback_action(my_position, map_state)
        return first, steps

    def _fallback_action(self, my_position, map_state):
        candidates = []
        for nxt, move in self._valid_moves(my_position, map_state):
            score = self._unknown_neighbors(nxt, self.known_map) * 4
            score -= self.visited_count.get(nxt, 0) * 2
            score += self.rng.random() * 0.1
            candidates.append((score, move))

        if not candidates:
            return Move.STAY, 1

        candidates.sort(reverse=True, key=lambda item: item[0])
        return candidates[0][1], 1


class GhostAgent(BaseGhostAgent, SearchMemoryMixin):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Codex Safety Hider"
        self._init_memory()

    def step(self, map_state, my_position, enemy_position, step_number):
        self._update_memory(map_state, my_position, enemy_position)
        threat = enemy_position or self.last_known_enemy_pos

        candidates = self._valid_moves(my_position, map_state, include_stay=True)
        if not candidates:
            return Move.STAY

        best_move = Move.STAY
        best_score = -10**9
        for nxt, move in candidates:
            score = self._score_ghost_position(nxt, move, threat, step_number)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _score_ghost_position(self, pos, move, threat, step_number):
        score = 0.0

        exits = len(self._valid_moves(pos, self.known_map))
        score += exits * 7

        if move == Move.STAY:
            score -= 8

        score -= self.visited_count.get(pos, 0) * 2.5
        if pos in list(self.previous_positions)[-3:]:
            score -= 6

        unknown_gain = self._unknown_neighbors(pos, self.known_map)
        score += unknown_gain * 1.5

        if threat is not None:
            maze_dist = self._maze_distance(pos, threat)
            straight_dist = self._manhattan(pos, threat)
            score += min(maze_dist, 18) * 5
            score += straight_dist * 1.5

            if straight_dist < 2:
                score -= 100
            elif straight_dist < 4:
                score -= 30

            if self._is_dead_end_like(pos) and straight_dist < 8:
                score -= 35
        else:
            center = (self.known_map.shape[0] // 2, self.known_map.shape[1] // 2)
            score += self._manhattan(pos, center) * 0.2

        if step_number % 11 == 0:
            score += self.rng.random()

        return score

    def _maze_distance(self, start, goal):
        if start == goal:
            return 0

        queue = deque([(start, 0)])
        visited = {start}
        while queue:
            current, dist = queue.popleft()
            if dist >= 28:
                break
            for nxt, _ in self._valid_moves(current, self.known_map):
                if nxt in visited:
                    continue
                if nxt == goal:
                    return dist + 1
                visited.add(nxt)
                queue.append((nxt, dist + 1))
        return 30 + self._manhattan(start, goal)

    def _is_dead_end_like(self, pos):
        exits = len(self._valid_moves(pos, self.known_map))
        if exits >= 3:
            return False
        distances = self._distance_map(pos, self.known_map, max_depth=5)
        junctions = 0
        for cell in distances:
            if len(self._valid_moves(cell, self.known_map)) >= 3:
                junctions += 1
        return junctions == 0
