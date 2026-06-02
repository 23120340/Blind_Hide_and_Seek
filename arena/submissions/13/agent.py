import sys
import os
import heapq
import random
from collections import deque, defaultdict
from typing import Optional, Tuple, List, Dict, Set

import numpy as np


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
src_dir = os.path.join(parent_dir, "src")

sys.path.append(parent_dir)
sys.path.append(src_dir)

try:
    from agent_interface import PacmanAgent as BasePacmanAgent
    from agent_interface import GhostAgent as BaseGhostAgent
    from environment import Move
except ImportError:
    try:
        from src.agent_interface import PacmanAgent as BasePacmanAgent
        from src.agent_interface import GhostAgent as BaseGhostAgent
        from src.environment import Move
    except ImportError:
        from agent_interface import PacmanAgent as BasePacmanAgent
        from agent_interface import GhostAgent as BaseGhostAgent
        from agent_interface import Move



# SYSTEM & UTILS

class Utils:
    WALL = 1
    EMPTY = 0
    UNKNOWN = -1

    DIRECTIONS = [
        Move.UP,
        Move.DOWN,
        Move.LEFT,
        Move.RIGHT
    ]

    MOVE_DELTA = {
        Move.UP: (-1, 0),
        Move.DOWN: (1, 0),
        Move.LEFT: (0, -1),
        Move.RIGHT: (0, 1),
        Move.STAY: (0, 0),
    }

    OPPOSITE = {
        Move.UP: Move.DOWN,
        Move.DOWN: Move.UP,
        Move.LEFT: Move.RIGHT,
        Move.RIGHT: Move.LEFT,
    }

    @staticmethod
    def is_within_bounds(pos, map_state):
        r, c = pos
        h, w = map_state.shape
        return 0 <= r < h and 0 <= c < w

    @staticmethod
    def manhattan_dist(p1, p2):
        return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

    @staticmethod
    def apply_move(pos, move, steps=1):
        dr, dc = Utils.MOVE_DELTA.get(move, (0, 0))
        return (pos[0] + dr * steps, pos[1] + dc * steps)

    @staticmethod
    def translate_pos_to_move(current_pos, next_pos):
        dr = next_pos[0] - current_pos[0]
        dc = next_pos[1] - current_pos[1]

        for move, delta in Utils.MOVE_DELTA.items():
            if delta == (dr, dc):
                return move

        return Move.STAY

    @staticmethod
    def is_walkable(pos, map_state, allow_unknown=False):
        if not Utils.is_within_bounds(pos, map_state):
            return False

        r, c = pos
        cell_value = int(map_state[r, c])

        if cell_value == Utils.WALL:
            return False

        if cell_value == Utils.UNKNOWN:
            return allow_unknown

        return True

    @staticmethod
    def get_valid_neighbors(pos, map_state, allow_unknown=False):
        neighbors = []

        for move in Utils.DIRECTIONS:
            next_pos = Utils.apply_move(pos, move)

            if Utils.is_walkable(next_pos, map_state, allow_unknown):
                neighbors.append((move, next_pos))

        return neighbors

    @staticmethod
    def is_straight_path_clear(start_pos, move, steps, map_state, allow_unknown=False):
        for s in range(1, steps + 1):
            check_pos = Utils.apply_move(start_pos, move, s)

            if not Utils.is_walkable(check_pos, map_state, allow_unknown):
                return False

        return True

    @staticmethod
    def max_clear_steps(start_pos, move, max_steps, map_state, allow_unknown=False):
        steps = 0

        for s in range(1, max_steps + 1):
            check_pos = Utils.apply_move(start_pos, move, s)

            if not Utils.is_walkable(check_pos, map_state, allow_unknown):
                break

            steps += 1

        return steps

    @staticmethod
    def is_same_direction(move1, move2):
        return move1 == move2

    @staticmethod
    def get_escape_count(pos, map_state, allow_unknown=False):
        return len(Utils.get_valid_neighbors(pos, map_state, allow_unknown))

    @staticmethod
    def all_walkable_cells(map_state, allow_unknown=True):
        h, w = map_state.shape
        cells = []

        for r in range(h):
            for c in range(w):
                pos = (r, c)

                if Utils.is_walkable(pos, map_state, allow_unknown):
                    cells.append(pos)

        return cells


MOVES = Utils.DIRECTIONS
MOVE_DELTA = Utils.MOVE_DELTA
OPPOSITE = Utils.OPPOSITE

WALL = Utils.WALL
EMPTY = Utils.EMPTY
UNKNOWN = Utils.UNKNOWN


def in_bounds(pos, map_state):
    return Utils.is_within_bounds(pos, map_state)


def is_walkable(pos, map_state, allow_unknown=False):
    return Utils.is_walkable(pos, map_state, allow_unknown)


def apply_move(pos, move, steps=1):
    return Utils.apply_move(pos, move, steps)


def manhattan_dist(p1, p2):
    return Utils.manhattan_dist(p1, p2)


def translate_pos_to_move(current_pos, next_pos):
    return Utils.translate_pos_to_move(current_pos, next_pos)


def get_valid_neighbors(pos, map_state, allow_unknown=False):
    return Utils.get_valid_neighbors(pos, map_state, allow_unknown)


def is_straight_path_clear(start_pos, move, steps, map_state, allow_unknown=False):
    return Utils.is_straight_path_clear(
        start_pos,
        move,
        steps,
        map_state,
        allow_unknown
    )


def max_clear_steps(start_pos, move, max_steps, map_state, allow_unknown=False):
    return Utils.max_clear_steps(
        start_pos,
        move,
        max_steps,
        map_state,
        allow_unknown
    )


def get_escape_count(pos, map_state, allow_unknown=False):
    return Utils.get_escape_count(pos, map_state, allow_unknown)


def all_walkable_cells(map_state, allow_unknown=True):
    return Utils.all_walkable_cells(map_state, allow_unknown)


def a_star(start, target, map_state, allow_unknown=True):
    if target is None:
        return []

    if not in_bounds(target, map_state):
        return []

    if start == target:
        return []

    if not is_walkable(start, map_state, allow_unknown=True):
        return []

    if not is_walkable(target, map_state, allow_unknown=allow_unknown):
        return []

    pq = []
    heapq.heappush(pq, (manhattan_dist(start, target), 0, start))

    parent = {start: None}
    g_score = {start: 0}

    while pq:
        _, g, cur = heapq.heappop(pq)

        if cur == target:
            path = []

            while cur != start:
                path.append(cur)
                cur = parent[cur]

            path.reverse()
            return path

        if g != g_score.get(cur, 10**9):
            continue

        for _, nxt in get_valid_neighbors(cur, map_state, allow_unknown=allow_unknown):
            ng = g + 1

            if ng < g_score.get(nxt, 10**9):
                g_score[nxt] = ng
                parent[nxt] = cur
                f = ng + manhattan_dist(nxt, target)
                heapq.heappush(pq, (f, ng, nxt))

    return []


def bfs_distances(start, map_state, allow_unknown=True):
    if start is None:
        return {}

    dist = {start: 0}
    q = deque([start])

    while q:
        cur = q.popleft()

        for _, nxt in get_valid_neighbors(cur, map_state, allow_unknown=allow_unknown):
            if nxt not in dist:
                dist[nxt] = dist[cur] + 1
                q.append(nxt)

    return dist



# SEEKER / PACMAN

class SeekerBrain:
    def __init__(self, pacman_speed=2):
        self.pacman_speed = max(1, int(pacman_speed))
        self.visited = defaultdict(int)
        self.prev_move = None

        self.last_known_enemy_pos = None
        self.last_seen_step = -1
        self.enemy_history = deque(maxlen=5)

        self.current_target = None
        self.target_hold_until = 0

        self.stuck_counter = 0
        self.last_positions = deque(maxlen=8)

    def compute_move_from_path(self, path, my_pos, map_state):
        if not path:
            return self.fallback_move(my_pos, map_state)

        move = translate_pos_to_move(my_pos, path[0])

        if move == Move.STAY:
            return (Move.STAY, 1)

        steps = 1

        for s in range(2, self.pacman_speed + 1):
            if s > len(path):
                break

            prev = my_pos if s == 1 else path[s - 2]
            curr = path[s - 1]

            if translate_pos_to_move(prev, curr) != move:
                break

            if not is_straight_path_clear(
                my_pos,
                move,
                s,
                map_state,
                allow_unknown=True
            ):
                break

            steps = s

        return (move, steps)

    def predict_enemy_position(self, map_state):
        if self.last_known_enemy_pos is None:
            return None

        if len(self.enemy_history) < 2:
            return self.last_known_enemy_pos

        old_pos = self.enemy_history[-2]
        new_pos = self.enemy_history[-1]

        dr = new_pos[0] - old_pos[0]
        dc = new_pos[1] - old_pos[1]

        predicted = (new_pos[0] + dr, new_pos[1] + dc)

        if in_bounds(predicted, map_state) and is_walkable(
            predicted,
            map_state,
            allow_unknown=True
        ):
            return predicted

        return self.last_known_enemy_pos

    def get_intercept_targets(self, enemy_pos, map_state):
        targets = []

        if enemy_pos is None:
            return targets

        targets.append(enemy_pos)

        for move in MOVES:
            for step in range(1, 4):
                pos = apply_move(enemy_pos, move, step)

                if is_walkable(pos, map_state, allow_unknown=True):
                    targets.append(pos)
                else:
                    break

        scored = []

        for pos in targets:
            score = get_escape_count(pos, map_state, allow_unknown=True)

            if int(map_state[pos[0], pos[1]]) == UNKNOWN:
                score += 2

            scored.append((score, pos))

        scored.sort(reverse=True, key=lambda x: x[0])

        return [pos for _, pos in scored]

    def chase_enemy(self, my_pos, enemy_pos, map_state, step_number):
        self.last_known_enemy_pos = enemy_pos
        self.last_seen_step = step_number
        self.enemy_history.append(enemy_pos)

        targets = []

        predicted = self.predict_enemy_position(map_state)

        if predicted is not None:
            targets.append(predicted)

        targets.extend(self.get_intercept_targets(enemy_pos, map_state))

        best_path = None
        best_score = -10**9

        for target in targets:
            path = a_star(
                my_pos,
                target,
                map_state,
                allow_unknown=True
            )

            if not path:
                continue

            path_len = len(path)
            dist_to_enemy = manhattan_dist(target, enemy_pos)
            escape = get_escape_count(target, map_state, allow_unknown=True)

            score = 0
            score -= path_len * 5
            score -= dist_to_enemy * 2
            score += escape * 0.5

            if target == enemy_pos:
                score += 20

            if score > best_score:
                best_score = score
                best_path = path

        if best_path:
            return self.compute_move_from_path(best_path, my_pos, map_state)

        return self.greedy_to_target(my_pos, enemy_pos, map_state)

    def score_frontier_cell(self, cell, my_pos, map_state):
        r, c = cell
        h, w = map_state.shape

        dist = manhattan_dist(my_pos, cell)
        visited_count = self.visited[cell]
        escape = get_escape_count(cell, map_state, allow_unknown=True)

        unknown_around = 0
        wall_around = 0
        empty_around = 0

        for move in MOVES:
            adj = apply_move(cell, move)

            if not in_bounds(adj, map_state):
                continue

            val = int(map_state[adj[0], adj[1]])

            if val == UNKNOWN:
                unknown_around += 1
            elif val == WALL:
                wall_around += 1
            elif val == EMPTY:
                empty_around += 1

        score = 0

        score += unknown_around * 12
        score += empty_around * 2
        score += escape * 4
        score += min(dist, 12) * 1.2

        score -= visited_count * 10
        score -= wall_around * 1.5

        if escape <= 1:
            score -= 15

        if self.prev_move is not None:
            first_move = translate_pos_to_move(my_pos, cell)

            if first_move == OPPOSITE.get(self.prev_move):
                score -= 5

        center_r, center_c = h // 2, w // 2
        center_dist = abs(r - center_r) + abs(c - center_c)
        score -= center_dist * 0.15

        if int(map_state[r, c]) == UNKNOWN:
            score += 10

        return score

    def find_best_frontier_target(self, my_pos, map_state):
        reachable = bfs_distances(
            my_pos,
            map_state,
            allow_unknown=True
        )

        candidates = []

        for pos in reachable:
            if pos == my_pos:
                continue

            r, c = pos
            val = int(map_state[r, c])

            unknown_around = 0

            for move in MOVES:
                adj = apply_move(pos, move)

                if in_bounds(adj, map_state):
                    if int(map_state[adj[0], adj[1]]) == UNKNOWN:
                        unknown_around += 1

            if val == UNKNOWN or unknown_around > 0:
                candidates.append(pos)

        if not candidates:
            return None

        return max(
            candidates,
            key=lambda p: self.score_frontier_cell(
                p,
                my_pos,
                map_state
            )
        )

    def choose_patrol_target(self, my_pos, map_state):
        h, w = map_state.shape

        patrol_points = [
            (h // 2, w // 2),
            (h // 2, w // 4),
            (h // 2, 3 * w // 4),
            (h // 4, w // 2),
            (3 * h // 4, w // 2),
            (h // 4, w // 4),
            (h // 4, 3 * w // 4),
            (3 * h // 4, w // 4),
            (3 * h // 4, 3 * w // 4),
        ]

        valid_points = [
            p for p in patrol_points
            if is_walkable(p, map_state, allow_unknown=True)
        ]

        if not valid_points:
            return None

        return max(
            valid_points,
            key=lambda p: (
                manhattan_dist(my_pos, p)
                - self.visited[p] * 5
                + get_escape_count(p, map_state, allow_unknown=True)
            )
        )

    def detect_stuck(self, my_pos):
        self.last_positions.append(my_pos)

        if len(self.last_positions) < self.last_positions.maxlen:
            return False

        unique_positions = set(self.last_positions)

        if len(unique_positions) <= 3:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0

        return self.stuck_counter >= 2

    def explore(self, my_pos, map_state, step_number):
        is_stuck = self.detect_stuck(my_pos)

        if (
            self.current_target is not None
            and step_number < self.target_hold_until
            and is_walkable(self.current_target, map_state, allow_unknown=True)
            and manhattan_dist(my_pos, self.current_target) > 1
            and not is_stuck
        ):
            path = a_star(
                my_pos,
                self.current_target,
                map_state,
                allow_unknown=True
            )

            if path:
                return self.compute_move_from_path(path, my_pos, map_state)

        target = self.find_best_frontier_target(my_pos, map_state)

        if target is None:
            target = self.choose_patrol_target(my_pos, map_state)

        if target is not None:
            self.current_target = target
            self.target_hold_until = step_number + 6

            path = a_star(
                my_pos,
                target,
                map_state,
                allow_unknown=True
            )

            if path:
                return self.compute_move_from_path(path, my_pos, map_state)

        return self.fallback_move(my_pos, map_state)

    def fallback_move(self, my_pos, map_state):
        candidates = []

        for move in MOVES:
            max_steps = max_clear_steps(
                my_pos,
                move,
                self.pacman_speed,
                map_state,
                allow_unknown=True
            )

            if max_steps <= 0:
                continue

            for steps in range(1, max_steps + 1):
                npos = apply_move(my_pos, move, steps)

                score = 0
                score += steps * 3
                score += get_escape_count(npos, map_state, allow_unknown=True) * 2
                score -= self.visited[npos] * 8

                if int(map_state[npos[0], npos[1]]) == UNKNOWN:
                    score += 12

                unknown_around = 0

                for m2 in MOVES:
                    adj = apply_move(npos, m2)

                    if in_bounds(adj, map_state):
                        if int(map_state[adj[0], adj[1]]) == UNKNOWN:
                            unknown_around += 1

                score += unknown_around * 5

                if self.prev_move is not None and move == OPPOSITE.get(self.prev_move):
                    score -= 6

                if get_escape_count(npos, map_state, allow_unknown=True) <= 1:
                    score -= 8

                candidates.append((score, move, steps))

        if not candidates:
            return (Move.STAY, 1)

        candidates.sort(reverse=True, key=lambda x: x[0])

        _, move, steps = candidates[0]
        return (move, steps)

    def greedy_to_target(self, my_pos, target, map_state):
        candidates = []

        for move in MOVES:
            max_steps = max_clear_steps(
                my_pos,
                move,
                self.pacman_speed,
                map_state,
                allow_unknown=True
            )

            if max_steps <= 0:
                continue

            for steps in range(1, max_steps + 1):
                npos = apply_move(my_pos, move, steps)

                score = 0
                score -= manhattan_dist(npos, target) * 10
                score += steps * 2
                score -= self.visited[npos] * 2

                if self.prev_move is not None and move == OPPOSITE.get(self.prev_move):
                    score -= 2

                candidates.append((score, move, steps))

        if not candidates:
            return (Move.STAY, 1)

        candidates.sort(reverse=True, key=lambda x: x[0])
        return (candidates[0][1], candidates[0][2])

    def hunt_last_known_position(self, my_pos, map_state, step_number):
        if self.last_known_enemy_pos is None:
            return None

        age = step_number - self.last_seen_step

        if age > 20:
            self.last_known_enemy_pos = None
            return None

        search_targets = [self.last_known_enemy_pos]

        for move in MOVES:
            for step in range(1, min(5, age + 2)):
                pos = apply_move(self.last_known_enemy_pos, move, step)

                if is_walkable(pos, map_state, allow_unknown=True):
                    search_targets.append(pos)
                else:
                    break

        best_path = None
        best_score = -10**9

        for target in search_targets:
            path = a_star(
                my_pos,
                target,
                map_state,
                allow_unknown=True
            )

            if not path:
                continue

            score = 0
            score -= len(path) * 3
            score += get_escape_count(target, map_state, allow_unknown=True) * 2
            score -= self.visited[target] * 4

            if score > best_score:
                best_score = score
                best_path = path

        if best_path:
            return self.compute_move_from_path(best_path, my_pos, map_state)

        return None

    def get_action(self, map_state, my_pos, enemy_pos, step_number):
        self.visited[my_pos] += 1

        if enemy_pos is not None:
            return self.chase_enemy(
                my_pos,
                enemy_pos,
                map_state,
                step_number
            )

        hunt_action = self.hunt_last_known_position(
            my_pos,
            map_state,
            step_number
        )

        if hunt_action is not None:
            return hunt_action

        return self.explore(
            my_pos,
            map_state,
            step_number
        )


# AGENT CHÍNH ARENA GỌI

class PacmanAgent(BasePacmanAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.name = "Integrated Strong Seeker"

        self.pacman_speed = max(
            1,
            int(kwargs.get("pacman_speed", kwargs.get("speed", 2)))
        )

        self.brain = SeekerBrain(self.pacman_speed)

    def step(self, map_state, my_position, enemy_position, step_number):
        action = self.brain.get_action(
            map_state,
            my_position,
            enemy_position,
            step_number
        )

        self.brain.prev_move = action[0] if isinstance(action, tuple) else action

        return action


# HIDER/GHOST

class Hider:
    def __init__(self):
        self.dead_ends = set()
        self.map_analyzed = False
        self.known_map = None

    # 1. Tính khoảng cách từ Seeker đến các ô có thể đi
    def bfs_from_seeker(self, my_pos, enemy_pos, map_state):
        if enemy_pos is None:
            return {}

        h, w = map_state.shape
        distances = {}
        q = deque([(enemy_pos[0], enemy_pos[1], 0)])
        visited = {enemy_pos}

        while q:
            r, c, d = q.popleft()
            distances[(r, c)] = d

            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and map_state[nr, nc] != 1 and (nr, nc) not in visited:
                    visited.add((nr, nc))
                    q.append((nr, nc, d + 1))

        return distances

    # 2. Xác định các ô ngõ cụt trong map
    def identify_dead_ends(self, map_state):
        h, w = map_state.shape
        dead = set()

        for i in range(h):
            for j in range(w):
                if map_state[i, j] == 0 or map_state[i, j] == -1:
                    cnt = 0
                    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                        ni, nj = i + dr, j + dc
                        if 0 <= ni < h and 0 <= nj < w and map_state[ni, nj] != 1:
                            cnt += 1
                    if cnt <= 1:
                        dead.add((i, j))
        return dead

    # 3. Chấm điểm độ an toàn của một ô
    def evaluate_safety(self, pos, enemy_pos, distances, map_state):
        if enemy_pos is None:
            return 0

        # Càng xa Seeker càng tốt
        distance_score = distances.get(pos, 0)

        # Tránh ngõ cụt
        penalty = -100 if pos in self.dead_ends else 0

        # Thưởng nếu vào ô chưa khám phá (-1)
        if map_state[pos[0], pos[1]] == -1:
            penalty += 50

        return distance_score + penalty

    # 4. Chọn hướng chạy trốn khi thấy Seeker
    def flee_enemy(self, my_pos, enemy_pos, map_state):
        distances = self.bfs_from_seeker(my_pos, enemy_pos, map_state)

        best_move = Move.STAY
        best_score = -10**9

        for move in MOVES:
            nr = my_pos[0] + move.value[0]
            nc = my_pos[1] + move.value[1]
            if 0 <= nr < 21 and 0 <= nc < 21 and map_state[nr, nc] != 1:
                score = self.evaluate_safety((nr, nc), enemy_pos, distances, map_state)
                if score > best_score:
                    best_score = score
                    best_move = move

        return best_move

    # 5. Chọn vị trí an toàn khi không thấy Seeker
    def hide_when_safe(self, my_pos, map_state, step_number):
        # Ưu tiên: lên trên → phải → trái → xuống
        for dr, dc, move in [(-1,0,Move.UP), (0,1,Move.RIGHT), (0,-1,Move.LEFT), (1,0,Move.DOWN)]:
            nr, nc = my_pos[0] + dr, my_pos[1] + dc
            if 0 <= nr < 21 and 0 <= nc < 21 and map_state[nr, nc] != 1:
                return move
        return Move.STAY

    # 6. Quyết định hành động cuối cùng của Hider
    def get_hider_action(self, map_state, my_pos, enemy_pos, step_number):
        # Lưu map đã thấy
        if self.known_map is None:
            self.known_map = map_state.copy()
        else:
            for i in range(21):
                for j in range(21):
                    if map_state[i, j] != -1:
                        self.known_map[i, j] = map_state[i, j]

        # Phân tích dead ends 1 lần
        if not self.map_analyzed:
            self.dead_ends = self.identify_dead_ends(self.known_map)
            self.map_analyzed = True

        if enemy_pos is not None:
            return self.flee_enemy(my_pos, enemy_pos, map_state)
        else:
            return self.hide_when_safe(my_pos, map_state, step_number)
        
class GhostAgent(BaseGhostAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.hider = Hider()

    def step(self, map_state, my_position, enemy_position, step_number):
        return self.hider.get_hider_action(map_state, my_position, enemy_position, step_number)