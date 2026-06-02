"""
Strategic Pacman and Ghost agents for partial observability.
"""

import sys
from pathlib import Path
from collections import deque
import heapq
import random   
import numpy as np

# Add src to path to import the interface
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move


MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

CAPTURE_DISTANCE = 2
DEFAULT_BELIEF_SAMPLE = 12
DEFAULT_MAX_BELIEF = 120
DEFAULT_SEARCH_DEPTH = 1
PACMAN_CHASE_WEIGHT = 12
PACMAN_FRONTIER_WEIGHT = 2
PACMAN_DEGREE_WEIGHT = 1.5
PACMAN_UNKNOWN_BONUS = 0.5
PACMAN_BACKTRACK_PENALTY = 4
GHOST_DISTANCE_WEIGHT = 12
GHOST_DEGREE_WEIGHT = 1.5
GHOST_UNKNOWN_BONUS = 0.5
GHOST_BACKTRACK_PENALTY = 4
GHOST_WORST_CASE_WEIGHT = 28
GHOST_IMMEDIATE_WEIGHT = 8
GHOST_REPLY_WEIGHT = 12
GHOST_SPACE_WEIGHT = 0.9
GHOST_STAY_PENALTY = 6
GHOST_DANGER_PENALTY = 1000
GHOST_LOCAL_SPACE_DEPTH = 4


def _apply_move(pos, move):
    delta_row, delta_col = move.value
    return (pos[0] + delta_row, pos[1] + delta_col)


def _in_bounds(pos, map_state):
    height, width = map_state.shape
    return 0 <= pos[0] < height and 0 <= pos[1] < width


def _is_passable(pos, map_state):
    if not _in_bounds(pos, map_state):
        return False
    return map_state[pos[0], pos[1]] == 0


def _neighbors(pos, map_state):
    result = []
    for move in MOVES:
        nxt = _apply_move(pos, move)
        if _is_passable(nxt, map_state):
            result.append((nxt, move))
    return result


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _visible_cells(map_state):
    coords = np.argwhere(map_state != -1)
    return {(int(r), int(c)) for r, c in coords}


def _adjacent_unknown_count(pos, map_state):
    count = 0
    for move in MOVES:
        nxt = _apply_move(pos, move)
        if _in_bounds(nxt, map_state) and map_state[nxt[0], nxt[1]] == -1:
            count += 1
    return count


def _min_distance(dist_map, targets, fallback):
    if not targets:
        return fallback
    best = fallback
    for target in targets:
        dist = dist_map[target]
        if dist >= 0 and dist < best:
            best = dist
    return best


def _is_frontier(pos, map_state):
    if not _is_passable(pos, map_state):
        return False
    for move in MOVES:
        nxt = _apply_move(pos, move)
        if _in_bounds(nxt, map_state) and map_state[nxt[0], nxt[1]] == -1:
            return True
    return False


def _frontier_distance_map(map_state):
    height, width = map_state.shape
    dist = np.full((height, width), -1, dtype=int)
    queue = deque()

    for r in range(height):
        for c in range(width):
            if _is_frontier((r, c), map_state):
                dist[r, c] = 0
                queue.append((r, c))

    while queue:
        cur = queue.popleft()
        for nxt, _ in _neighbors(cur, map_state):
            if dist[nxt] != -1:
                continue
            dist[nxt] = dist[cur] + 1
            queue.append(nxt)

    return dist


def _reconstruct_path(came_from, move_from, start, goal):
    path = []
    cur = goal
    while cur != start:
        move = move_from[cur]
        path.append(move)
        cur = came_from[cur]
    path.reverse()
    return path


def _bfs_path(start, goal, map_state):
    if start == goal:
        return []
    queue = deque([start])
    came_from = {start: None}
    move_from = {}

    while queue:
        cur = queue.popleft()
        for nxt, move in _neighbors(cur, map_state):
            if nxt in came_from:
                continue
            came_from[nxt] = cur
            move_from[nxt] = move
            if nxt == goal:
                return _reconstruct_path(came_from, move_from, start, goal)
            queue.append(nxt)
    return None


def _a_star_path(start, goal, map_state):
    if start == goal:
        return []
    if not _is_passable(start, map_state) or not _is_passable(goal, map_state):
        return None

    frontier = []
    heapq.heappush(frontier, (0, start))
    came_from = {start: None}
    move_from = {}
    cost_so_far = {start: 0}

    while frontier:
        _, cur = heapq.heappop(frontier)
        if cur == goal:
            return _reconstruct_path(came_from, move_from, start, goal)

        for nxt, move in _neighbors(cur, map_state):
            new_cost = cost_so_far[cur] + 1
            old_cost = cost_so_far.get(nxt)
            if old_cost is not None and new_cost >= old_cost:
                continue
            cost_so_far[nxt] = new_cost
            priority = new_cost + _manhattan(nxt, goal)
            heapq.heappush(frontier, (priority, nxt))
            came_from[nxt] = cur
            move_from[nxt] = move

    return None


def _bfs_distances(start, map_state):
    height, width = map_state.shape
    dist = np.full((height, width), -1, dtype=int)
    if not _is_passable(start, map_state):
        return dist

    queue = deque([start])
    dist[start] = 0
    while queue:
        cur = queue.popleft()
        for nxt, _ in _neighbors(cur, map_state):
            if dist[nxt] != -1:
                continue
            dist[nxt] = dist[cur] + 1
            queue.append(nxt)
    return dist


def _reachable_within(start, map_state, max_depth):
    if max_depth <= 0 or not _is_passable(start, map_state):
        return 0

    visited = {start}
    queue = deque([(start, 0)])
    while queue:
        cur, depth = queue.popleft()
        if depth >= max_depth:
            continue
        for nxt, _ in _neighbors(cur, map_state):
            if nxt in visited:
                continue
            visited.add(nxt)
            queue.append((nxt, depth + 1))
    return len(visited)


def _path_to_nearest(start, map_state, predicate):
    queue = deque([start])
    came_from = {start: None}
    move_from = {}

    while queue:
        cur = queue.popleft()
        if predicate(cur):
            return _reconstruct_path(came_from, move_from, start, cur)
        for nxt, move in _neighbors(cur, map_state):
            if nxt in came_from:
                continue
            came_from[nxt] = cur
            move_from[nxt] = move
            queue.append(nxt)
    return None


def _path_to_frontier(start, map_state):
    return _path_to_nearest(start, map_state, lambda pos: _is_frontier(pos, map_state))


def _max_straight_steps(pos, move, map_state, max_steps):
    steps = 0
    current = pos
    for _ in range(max_steps):
        nxt = _apply_move(current, move)
        if not _is_passable(nxt, map_state):
            break
        steps += 1
        current = nxt
    return steps


def _apply_pacman_action(pos, move, steps, map_state):
    if move == Move.STAY:
        return pos

    current = pos
    for _ in range(steps):
        nxt = _apply_move(current, move)
        if not _is_passable(nxt, map_state):
            break
        current = nxt
    return current


def _pacman_actions(pos, map_state, max_speed):
    actions = []
    for move in MOVES:
        max_steps = _max_straight_steps(pos, move, map_state, max_speed)
        if max_steps <= 0:
            continue
        actions.append((move, 1))
        if max_steps > 1:
            actions.append((move, max_steps))
    if not actions:
        actions = [(Move.STAY, 1)]
    return actions


def _ghost_actions(pos, map_state):
    actions = [Move.STAY]
    for move in MOVES:
        nxt = _apply_move(pos, move)
        if _is_passable(nxt, map_state):
            actions.append(move)
    return actions


def _sample_positions(positions, limit, rng):
    if not positions:
        return []
    if len(positions) <= limit:
        return list(positions)
    return rng.sample(list(positions), limit)


class PacmanAgent(BasePacmanAgent):
    """
    Pacman agent that uses frontier exploration and expectiminimax under fog.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Strategic Pacman"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.search_depth = max(1, int(kwargs.get("search_depth", DEFAULT_SEARCH_DEPTH)))
        self.belief_sample_size = max(1, int(kwargs.get("belief_sample_size", DEFAULT_BELIEF_SAMPLE)))
        self.max_belief_size = max(1, int(kwargs.get("max_belief_size", DEFAULT_MAX_BELIEF)))
        self.memory_horizon = max(1, int(kwargs.get("memory_horizon", 12)))
        self.enemy_belief = set()
        self.last_seen_step = -9999
        self.known_map = None
        self.frontier_dist_map = None
        self.recent_positions = deque(maxlen=6)
        self.last_move = None
        self.rng = random.Random()
        self._dist_cache = {}
        self.visible_enemy = None
        self.current_step = 0
        self.last_known_enemy_pos = None

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int):
        self.current_step = step_number
        self.recent_positions.append(my_position)
        self.visible_enemy = enemy_position
        self._dist_cache = {}
        self._update_known_map(map_state)
        self._update_enemy_belief(map_state, enemy_position, step_number)
        self.frontier_dist_map = _frontier_distance_map(self.known_map)

        action = self._choose_action(my_position)
        self.last_move = action[0]
        return action

    def _update_known_map(self, map_state):
        if self.known_map is None:
            self.known_map = map_state.copy()
            return

        mask = map_state != -1
        self.known_map[mask] = map_state[mask]

    def _init_belief(self, visible_cells):
        empty_cells = np.argwhere(self.known_map == 0)
        return {(int(r), int(c)) for r, c in empty_cells if (int(r), int(c)) not in visible_cells}

    def _expand_belief(self, belief):
        expanded = set()
        for pos in belief:
            expanded.add(pos)
            for move in MOVES:
                nxt = _apply_move(pos, move)
                if _is_passable(nxt, self.known_map):
                    expanded.add(nxt)
        return expanded

    def _cap_belief(self, belief):
        if len(belief) <= self.max_belief_size:
            return belief
        return set(self.rng.sample(list(belief), self.max_belief_size))

    def _update_enemy_belief(self, map_state, enemy_position, step_number):
        visible_cells = _visible_cells(map_state)

        if enemy_position is not None:
            self.enemy_belief = {enemy_position}
            self.last_seen_step = step_number
            self.last_known_enemy_pos = enemy_position
        else:
            stale = (self.last_seen_step == -9999 or
                     step_number - self.last_seen_step > self.memory_horizon)
            if stale:
                self.last_known_enemy_pos = None
                self.enemy_belief = self._init_belief(visible_cells)
            else:
                self.enemy_belief = self._expand_belief(self.enemy_belief)
                self.enemy_belief.difference_update(visible_cells)
                if not self.enemy_belief:
                    self.enemy_belief = self._init_belief(visible_cells)

        self.enemy_belief = self._cap_belief(self.enemy_belief)

    def _distance_map(self, start):
        cached = self._dist_cache.get(start)
        if cached is not None:
            return cached
        dist = _bfs_distances(start, self.known_map)
        self._dist_cache[start] = dist
        return dist

    def _evaluate_pacman(self, pacman_pos, ghost_positions):
        if ghost_positions:
            dist_map = self._distance_map(pacman_pos)
            min_dist = _min_distance(dist_map, ghost_positions, 50)
            if min_dist < CAPTURE_DISTANCE:
                return 1000
        else:
            min_dist = 0

        frontier_dist = 0
        if self.frontier_dist_map is not None:
            frontier_dist = int(self.frontier_dist_map[pacman_pos])
            if frontier_dist < 0:
                frontier_dist = 0

        degree = len(_neighbors(pacman_pos, self.known_map))
        unknown_adj = _adjacent_unknown_count(pacman_pos, self.known_map)
        backtrack = 1 if pacman_pos in self.recent_positions else 0
        return (
            -min_dist * PACMAN_CHASE_WEIGHT
            -frontier_dist * PACMAN_FRONTIER_WEIGHT
            +degree * PACMAN_DEGREE_WEIGHT
            +unknown_adj * PACMAN_UNKNOWN_BONUS
            -backtrack * PACMAN_BACKTRACK_PENALTY
        )

    def _expect_value(self, pacman_pos, ghost_positions, depth):
        if depth <= 0 or not ghost_positions:
            return self._evaluate_pacman(pacman_pos, ghost_positions)

        total = 0.0
        for ghost_pos in ghost_positions:
            total += self._min_value(pacman_pos, ghost_pos, depth)
        return total / len(ghost_positions)

    def _max_value(self, pacman_pos, ghost_pos, depth):
        if depth <= 0 or _manhattan(pacman_pos, ghost_pos) < CAPTURE_DISTANCE:
            return self._evaluate_pacman(pacman_pos, {ghost_pos})

        best = -1e9
        for action in _pacman_actions(pacman_pos, self.known_map, self.pacman_speed):
            new_pos = _apply_pacman_action(pacman_pos, action[0], action[1], self.known_map)
            score = self._min_value(new_pos, ghost_pos, depth)
            if score > best:
                best = score
        return best

    def _min_value(self, pacman_pos, ghost_pos, depth):
        if _manhattan(pacman_pos, ghost_pos) < CAPTURE_DISTANCE:
            return 1000

        best = 1e9
        for move in _ghost_actions(ghost_pos, self.known_map):
            new_pos = ghost_pos if move == Move.STAY else _apply_move(ghost_pos, move)
            score = self._max_value(pacman_pos, new_pos, depth - 1)
            if score < best:
                best = score
        return best

    def _chase_target_action(self, my_position, target):
        path = _a_star_path(my_position, target, self.known_map)
        if path is None:
            path = _bfs_path(my_position, target, self.known_map)
        if path is None:
            return None
        return self._path_to_action(path)

    def _nearest_reachable_target(self, my_position, targets):
        if not targets:
            return None

        dist_map = self._distance_map(my_position)
        best_target = None
        best_dist = 10**9
        for target in targets:
            dist = dist_map[target]
            if dist >= 0 and dist < best_dist:
                best_dist = dist
                best_target = target
        return best_target

    def _choose_action(self, my_position):
        actions = _pacman_actions(my_position, self.known_map, self.pacman_speed)
        belief_sample = _sample_positions(self.enemy_belief, self.belief_sample_size, self.rng)

        if self.visible_enemy is not None:
            chase_action = self._chase_target_action(my_position, self.visible_enemy)
            if chase_action is not None:
                return chase_action

        if self.last_known_enemy_pos is not None:
            if my_position == self.last_known_enemy_pos:
                self.last_known_enemy_pos = None
            else:
                chase_action = self._chase_target_action(my_position, self.last_known_enemy_pos)
                if chase_action is not None:
                    return chase_action

        belief_target = self._nearest_reachable_target(my_position, self.enemy_belief)
        if belief_target is not None:
            chase_action = self._chase_target_action(my_position, belief_target)
            if chase_action is not None:
                return chase_action

        if not belief_sample:
            path = _path_to_frontier(my_position, self.known_map)
            if path:
                return self._path_to_action(path)
            return self._fallback_move(self.known_map, my_position)

        best_score = -1e9
        best_action = actions[0]
        for action in actions:
            new_pos = _apply_pacman_action(my_position, action[0], action[1], self.known_map)
            score = self._expect_value(new_pos, belief_sample, self.search_depth)
            if score > best_score:
                best_score = score
                best_action = action

        return best_action

    def _path_to_action(self, path):
        if not path:
            return (Move.STAY, 1)

        move = path[0]
        steps = 1
        while (steps < self.pacman_speed and
               steps < len(path) and
               path[steps] == move):
            steps += 1
        return (move, steps)

    def _fallback_move(self, map_state, my_position):
        valid_moves = [m for m in MOVES if _is_passable(_apply_move(my_position, m), map_state)]
        if not valid_moves:
            return (Move.STAY, 1)

        best_move = None
        best_score = -1e9
        for move in valid_moves:
            new_pos = _apply_move(my_position, move)
            degree = len(_neighbors(new_pos, map_state))
            backtrack = 1 if new_pos in self.recent_positions else 0
            score = degree * 2 - backtrack * 4
            if self.last_move == move:
                score += 1
            score += self.rng.random() * 0.01

            if score > best_score:
                best_score = score
                best_move = move

        steps = _max_straight_steps(my_position, best_move, map_state, self.pacman_speed)
        steps = max(1, steps)
        self.last_move = best_move
        return (best_move, steps)


class GhostAgent(BaseGhostAgent):
    """
    Ghost agent that uses belief search to maximize distance from Pacman.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Strategic Ghost"
        self.search_depth = max(1, int(kwargs.get("search_depth", max(2, DEFAULT_SEARCH_DEPTH))))
        self.belief_sample_size = max(1, int(kwargs.get("belief_sample_size", DEFAULT_BELIEF_SAMPLE)))
        self.max_belief_size = max(1, int(kwargs.get("max_belief_size", DEFAULT_MAX_BELIEF)))
        self.memory_horizon = max(1, int(kwargs.get("memory_horizon", 12)))
        self.enemy_speed = max(1, int(kwargs.get("enemy_speed", 2)))
        self.enemy_belief = set()
        self.last_seen_step = -9999
        self.known_map = None
        self.recent_positions = deque(maxlen=6)
        self.last_move = None
        self.rng = random.Random()
        self._dist_cache = {}
        self.visible_enemy = None

    def step(self, map_state: np.ndarray,
             my_position: tuple,
             enemy_position: tuple,
             step_number: int) -> Move:
        self.recent_positions.append(my_position)
        self.visible_enemy = enemy_position
        self._dist_cache = {}
        self._update_known_map(map_state)
        self._update_enemy_belief(map_state, enemy_position, step_number)

        move = self._choose_action(my_position)
        self.last_move = move if move != Move.STAY else self.last_move
        return move

    def _update_known_map(self, map_state):
        if self.known_map is None:
            self.known_map = map_state.copy()
            return

        mask = map_state != -1
        self.known_map[mask] = map_state[mask]

    def _init_belief(self, visible_cells):
        empty_cells = np.argwhere(self.known_map == 0)
        return {(int(r), int(c)) for r, c in empty_cells if (int(r), int(c)) not in visible_cells}

    def _expand_belief(self, belief):
        expanded = set()
        for pos in belief:
            expanded.add(pos)
            for move in MOVES:
                steps = 0
                current = pos
                while steps < self.enemy_speed:
                    nxt = _apply_move(current, move)
                    if not _is_passable(nxt, self.known_map):
                        break
                    expanded.add(nxt)
                    current = nxt
                    steps += 1
        return expanded

    def _cap_belief(self, belief):
        if len(belief) <= self.max_belief_size:
            return belief
        return set(self.rng.sample(list(belief), self.max_belief_size))

    def _update_enemy_belief(self, map_state, enemy_position, step_number):
        visible_cells = _visible_cells(map_state)

        if enemy_position is not None:
            self.enemy_belief = {enemy_position}
            self.last_seen_step = step_number
        else:
            stale = (self.last_seen_step == -9999 or
                     step_number - self.last_seen_step > self.memory_horizon)
            if stale:
                self.enemy_belief = self._init_belief(visible_cells)
            else:
                self.enemy_belief = self._expand_belief(self.enemy_belief)
                self.enemy_belief.difference_update(visible_cells)
                if not self.enemy_belief:
                    self.enemy_belief = self._init_belief(visible_cells)

        self.enemy_belief = self._cap_belief(self.enemy_belief)

    def _distance_map(self, start):
        cached = self._dist_cache.get(start)
        if cached is not None:
            return cached
        dist = _bfs_distances(start, self.known_map)
        self._dist_cache[start] = dist
        return dist

    def _evaluate_ghost(self, ghost_pos, pacman_positions):
        if pacman_positions:
            dist_map = self._distance_map(ghost_pos)
            min_dist = _min_distance(dist_map, pacman_positions, 50)
            if min_dist < CAPTURE_DISTANCE:
                return -1000
        else:
            min_dist = 0

        degree = len(_neighbors(ghost_pos, self.known_map))
        unknown_adj = _adjacent_unknown_count(ghost_pos, self.known_map)
        backtrack = 1 if ghost_pos in self.recent_positions else 0
        return (
            min_dist * GHOST_DISTANCE_WEIGHT
            +degree * GHOST_DEGREE_WEIGHT
            +unknown_adj * GHOST_UNKNOWN_BONUS
            -backtrack * GHOST_BACKTRACK_PENALTY
        )

    def _escape_move(self, my_position, pacman_pos):
        best_move = Move.STAY
        best_score = -1e9
        for move in _ghost_actions(my_position, self.known_map):
            new_pos = my_position if move == Move.STAY else _apply_move(my_position, move)
            score = self._escape_score(new_pos, pacman_pos, move)
            score += 0.2 * self._expect_value(new_pos, {pacman_pos}, 1)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _pacman_reachable_positions(self, pacman_pos):
        reachable = {pacman_pos}
        for action in _pacman_actions(pacman_pos, self.known_map, self.enemy_speed):
            nxt = _apply_pacman_action(pacman_pos, action[0], action[1], self.known_map)
            reachable.add(nxt)
        return reachable

    def _distance_or_fallback(self, dist_map, start, target):
        dist = dist_map[target]
        if dist >= 0:
            return dist
        return _manhattan(start, target) + 8

    def _escape_score(self, ghost_pos, pacman_pos, move):
        dist_map = self._distance_map(ghost_pos)
        immediate_dist = self._distance_or_fallback(dist_map, ghost_pos, pacman_pos)
        worst_future_dist = 50
        for pacman_next in self._pacman_reachable_positions(pacman_pos):
            dist = self._distance_or_fallback(dist_map, ghost_pos, pacman_next)
            if dist < worst_future_dist:
                worst_future_dist = dist
        worst_reply_dist = self._worst_case_reply_distance(ghost_pos, pacman_pos)

        danger_penalty = GHOST_DANGER_PENALTY if worst_future_dist < CAPTURE_DISTANCE else 0
        if worst_reply_dist < CAPTURE_DISTANCE:
            danger_penalty += int(GHOST_DANGER_PENALTY * 0.5)
        degree = len(_neighbors(ghost_pos, self.known_map))
        unknown_adj = _adjacent_unknown_count(ghost_pos, self.known_map)
        backtrack = 1 if ghost_pos in self.recent_positions else 0
        local_space = _reachable_within(ghost_pos, self.known_map, GHOST_LOCAL_SPACE_DEPTH)
        stay_penalty = GHOST_STAY_PENALTY if move == Move.STAY else 0
        return (
            worst_future_dist * GHOST_WORST_CASE_WEIGHT
            + immediate_dist * GHOST_IMMEDIATE_WEIGHT
            + worst_reply_dist * GHOST_REPLY_WEIGHT
            + degree * GHOST_DEGREE_WEIGHT
            + unknown_adj * GHOST_UNKNOWN_BONUS
            + local_space * GHOST_SPACE_WEIGHT
            - backtrack * GHOST_BACKTRACK_PENALTY
            - stay_penalty
            - danger_penalty
        )

    def _worst_case_reply_distance(self, ghost_pos, pacman_pos):
        pacman_next_positions = self._pacman_reachable_positions(pacman_pos)
        worst_reply = 50
        for pacman_next in pacman_next_positions:
            best_reply = -1
            for move in _ghost_actions(ghost_pos, self.known_map):
                ghost_next = ghost_pos if move == Move.STAY else _apply_move(ghost_pos, move)
                dist_map = self._distance_map(ghost_next)
                dist = self._distance_or_fallback(dist_map, ghost_next, pacman_next)
                if dist > best_reply:
                    best_reply = dist
            if best_reply < worst_reply:
                worst_reply = best_reply
        return worst_reply

    def _expect_value(self, ghost_pos, pacman_positions, depth):
        if depth <= 0 or not pacman_positions:
            return self._evaluate_ghost(ghost_pos, pacman_positions)

        total = 0.0
        for pacman_pos in pacman_positions:
            total += self._min_value(ghost_pos, pacman_pos, depth)
        return total / len(pacman_positions)

    def _max_value(self, ghost_pos, pacman_pos, depth):
        if depth <= 0 or _manhattan(ghost_pos, pacman_pos) < CAPTURE_DISTANCE:
            return self._evaluate_ghost(ghost_pos, {pacman_pos})

        best = -1e9
        for move in _ghost_actions(ghost_pos, self.known_map):
            new_pos = ghost_pos if move == Move.STAY else _apply_move(ghost_pos, move)
            score = self._min_value(new_pos, pacman_pos, depth)
            if score > best:
                best = score
        return best

    def _min_value(self, ghost_pos, pacman_pos, depth):
        if _manhattan(ghost_pos, pacman_pos) < CAPTURE_DISTANCE:
            return -1000

        best = 1e9
        for action in _pacman_actions(pacman_pos, self.known_map, self.enemy_speed):
            new_pos = _apply_pacman_action(pacman_pos, action[0], action[1], self.known_map)
            score = self._max_value(ghost_pos, new_pos, depth - 1)
            if score < best:
                best = score
        return best

    def _choose_action(self, my_position):
        actions = _ghost_actions(my_position, self.known_map)
        belief_sample = _sample_positions(self.enemy_belief, self.belief_sample_size, self.rng)

        if self.visible_enemy is not None:
            return self._escape_move(my_position, self.visible_enemy)

        if not belief_sample:
            return self._patrol_move(self.known_map, my_position)

        best_score = -1e9
        best_move = actions[0]
        for move in actions:
            new_pos = my_position if move == Move.STAY else _apply_move(my_position, move)
            safety_score = 0.0
            for pacman_pos in belief_sample:
                safety_score += self._escape_score(new_pos, pacman_pos, move)
            safety_score /= len(belief_sample)
            score = 0.65 * safety_score + 0.35 * self._expect_value(new_pos, belief_sample, self.search_depth)
            if score > best_score:
                best_score = score
                best_move = move

        return best_move

    def _patrol_move(self, map_state, my_position):
        valid_moves = [m for m in MOVES if _is_passable(_apply_move(my_position, m), map_state)]
        if not valid_moves:
            return Move.STAY

        best_move = None
        best_score = -1e9
        for move in valid_moves:
            new_pos = _apply_move(my_position, move)
            degree = len(_neighbors(new_pos, map_state))
            backtrack = 1 if new_pos in self.recent_positions else 0
            local_space = _reachable_within(new_pos, map_state, GHOST_LOCAL_SPACE_DEPTH)
            score = degree * 2 + local_space * 0.7 - backtrack * 4
            if self.last_move == move:
                score += 1
            score += self.rng.random() * 0.01

            if score > best_score:
                best_score = score
                best_move = move

        return best_move
