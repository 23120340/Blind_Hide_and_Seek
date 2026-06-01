import sys
import time
import math
from pathlib import Path
from collections import deque

# Framework imports
src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move
import numpy as np

# Constants & Time Controller

TIME_BUDGET = 0.78       # 780ms usable out of 1000ms

ALL_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT, Move.STAY]
DIR_MOVES = [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]

_MOVE_DELTAS = {
    Move.UP: (-1, 0),
    Move.DOWN: (1, 0),
    Move.LEFT: (0, -1),
    Move.RIGHT: (0, 1),
    Move.STAY: (0, 0),
}


class TimeController:
    """Strict wall-clock timer to prevent timeout."""
    __slots__ = ('_start', '_budget')

    def __init__(self, budget_seconds: float):
        self._start = time.perf_counter()
        self._budget = budget_seconds

    def elapsed(self) -> float:
        return time.perf_counter() - self._start

    def remaining(self) -> float:
        return self._budget - self.elapsed()

    def has_time(self, min_seconds: float = 0.05) -> bool:
        return self.remaining() > min_seconds


class _TimeoutSignal(Exception):
    """Internal signal to abort deep search branches when time is up."""
    pass


# Shared Utilities

def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def apply_move(pos, move):
    dr, dc = _MOVE_DELTAS[move]
    return (pos[0] + dr, pos[1] + dc)


def is_passable(pos, known_map):
    r, c = pos
    h, w = known_map.shape
    if r < 0 or r >= h or c < 0 or c >= w:
        return False
    return known_map[r, c] != 1


def get_neighbors(pos, known_map):
    """Return list of (next_pos, move) for valid directional moves."""
    result = []
    for move in DIR_MOVES:
        np_ = apply_move(pos, move)
        if is_passable(np_, known_map):
            result.append((np_, move))
    return result


def bfs_shortest_path(start, goal, known_map):
    """BFS from start to goal. Returns list of Moves."""
    if start == goal:
        return []
    visited = {start}
    queue = deque([(start, [])])
    while queue:
        pos, path = queue.popleft()
        for npos, move in get_neighbors(pos, known_map):
            if npos in visited:
                continue
            new_path = path + [move]
            if npos == goal:
                return new_path
            visited.add(npos)
            queue.append((npos, new_path))
    return []


def safe_shortest_path(start, goal, known_map, threat_pos):
    """
    Threat-aware pathfinding using Dijkstra.
    Penalizes paths that get dangerously close to the threat (Pacman).
    """
    if start == goal:
        return []
        
    import heapq
    # Priority Queue elements: (cost, current_pos, path)
    pq = [(0, start, [])]
    visited = {start: 0}  # pos -> cost to reach
    
    while pq:
        cost, pos, path = heapq.heappop(pq)
        
        if pos == goal:
            return path
            
        if visited.get(pos, float('inf')) < cost:
            continue
            
        for npos, move in get_neighbors(pos, known_map):
            # Calculate step penalty based on threat proximity
            penalty = 1
            if threat_pos is not None:
                d = manhattan(npos, threat_pos)
                if d < 3:
                    penalty += 1000  # Extreme threat danger zone
                elif d < 5:
                    penalty += 100   # Avoid proximity
                    
            next_cost = cost + penalty
            if next_cost < visited.get(npos, float('inf')):
                visited[npos] = next_cost
                heapq.heappush(pq, (next_cost, npos, path + [move]))
                
    # Fallback to standard BFS if safe path is blocked
    return bfs_shortest_path(start, goal, known_map)



def bfs_distances(start, known_map):
    """BFS flood fill. Returns dict {pos: distance}."""
    distances = {start: 0}
    queue = deque([(start, 0)])
    while queue:
        pos, dist = queue.popleft()
        for npos, _ in get_neighbors(pos, known_map):
            if npos not in distances:
                distances[npos] = dist + 1
                queue.append((npos, dist + 1))
    return distances


def bfs_distances_limited(start, known_map, max_dist=10):
    """BFS flood fill limited to max_dist. Faster for local analysis."""
    distances = {start: 0}
    queue = deque([(start, 0)])
    while queue:
        pos, dist = queue.popleft()
        if dist >= max_dist:
            continue
        for npos, _ in get_neighbors(pos, known_map):
            if npos not in distances:
                distances[npos] = dist + 1
                queue.append((npos, dist + 1))
    return distances


def bfs_to_nearest_unseen(start, known_map):
    """BFS toward nearest unseen cell. Returns first Move or None."""
    visited = {start}
    queue = deque([(start, None)])
    while queue:
        pos, first_move = queue.popleft()
        for npos, move in get_neighbors(pos, known_map):
            if npos in visited:
                continue
            visited.add(npos)
            fm = first_move if first_move is not None else move
            r, c = npos
            if 0 <= r < known_map.shape[0] and 0 <= c < known_map.shape[1]:
                if known_map[r, c] == -1:
                    return fm
            queue.append((npos, fm))
    return None


def apply_pacman_move(pos, move, steps, known_map):
    """Simulate Pacman moving `steps` times in direction `move`."""
    if move == Move.STAY:
        return pos
    current = pos
    dr, dc = _MOVE_DELTAS[move]
    for _ in range(steps):
        npos = (current[0] + dr, current[1] + dc)
        if not is_passable(npos, known_map):
            break
        current = npos
    return current


def max_valid_steps(pos, move, known_map, max_steps):
    """How many valid steps in direction `move`."""
    if move == Move.STAY:
        return 0
    steps = 0
    current = pos
    dr, dc = _MOVE_DELTAS[move]
    for _ in range(max_steps):
        npos = (current[0] + dr, current[1] + dc)
        if not is_passable(npos, known_map):
            break
        steps += 1
        current = npos
    return steps


def get_visible_cells_cross(pos, radius, known_map):
    """Compute visible cells using cross LOS."""
    if radius <= 0:
        h, w = known_map.shape
        return {(r, c) for r in range(h) for c in range(w)}
    visible = {pos}
    row, col = pos
    h, w = known_map.shape
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for dist in range(1, radius + 1):
            nr, nc = row + dr * dist, col + dc * dist
            if nr < 0 or nr >= h or nc < 0 or nc >= w:
                break
            if known_map[nr, nc] == 1:
                break
            visible.add((nr, nc))
    return visible


def count_escape_routes(pos, pac_bfs, nav_map):
    """Count directions where Ghost can flee away from Pacman."""
    pac_d_here = pac_bfs.get(pos, 0)
    routes = 0
    for move in DIR_MOVES:
        npos = apply_move(pos, move)
        if is_passable(npos, nav_map):
            pac_d_there = pac_bfs.get(npos, 0)
            if pac_d_there > pac_d_here:
                routes += 1
    return routes


# Pre-computed Map Data

class MapData:
    """Pre-computed map analysis for O(1) evaluation lookups."""

    def __init__(self):
        self.mobility = None       # dict: pos -> neighbor_count
        self.dead_ends = None      # set: cells with <= 1 neighbor
        self.corridors = None      # set: cells with <= 2 neighbors
        self.junctions = None      # set: cells with >= 3 neighbors
        self._version = -1

    def update(self, known_map, version):
        if version == self._version:
            return
        self._version = version
        h, w = known_map.shape
        self.mobility = {}
        self.dead_ends = set()
        self.corridors = set()
        self.junctions = set()
        for r in range(h):
            for c in range(w):
                if known_map[r, c] == 1:
                    continue
                count = 0
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and known_map[nr, nc] != 1:
                        count += 1
                self.mobility[(r, c)] = count
                if count <= 1:
                    self.dead_ends.add((r, c))
                if count <= 2:
                    self.corridors.add((r, c))
                if count >= 3:
                    self.junctions.add((r, c))


# Shared MapData instance
_map_data = MapData()


# Transposition Table

TT_EXACT = 0
TT_LOWER = 1  # Failed low → score is a lower bound
TT_UPPER = 2  # Failed high → score is an upper bound


class TranspositionTable:
    """Hash table for Minimax search states."""
    __slots__ = ('_table', '_max_size')

    def __init__(self, max_size=100000):
        self._table = {}
        self._max_size = max_size

    def _key(self, pos1, pos2, is_max):
        """Unique integer key for (pos1, pos2, is_max) on 21x21 grid."""
        return (pos1[0] * 18522 + pos1[1] * 882
                + pos2[0] * 42 + pos2[1] * 2
                + (1 if is_max else 0))

    def lookup(self, pos1, pos2, depth, alpha, beta, is_max):
        """Try to use a stored result. Returns (score, best_move) or None."""
        key = self._key(pos1, pos2, is_max)
        entry = self._table.get(key)
        if entry is None:
            return None
        tt_depth, tt_score, tt_flag, tt_move = entry
        if tt_depth < depth:
            return None  # Not deep enough
        if tt_flag == TT_EXACT:
            return (tt_score, tt_move)
        elif tt_flag == TT_LOWER and tt_score >= beta:
            return (tt_score, tt_move)
        elif tt_flag == TT_UPPER and tt_score <= alpha:
            return (tt_score, tt_move)
        return None

    def get_best_move(self, pos1, pos2, is_max):
        """Get TT's best move hint for move ordering."""
        key = self._key(pos1, pos2, is_max)
        entry = self._table.get(key)
        if entry:
            return entry[3]
        return None

    def store(self, pos1, pos2, depth, score, flag, best_move, is_max):
        if len(self._table) >= self._max_size:
            self._table.clear()
        key = self._key(pos1, pos2, is_max)
        existing = self._table.get(key)
        if existing is None or existing[0] <= depth:
            self._table[key] = (depth, score, flag, best_move)

    def clear(self):
        self._table.clear()


# Belief State Tracker

class BeliefState:
    """Tracks probability distribution of enemy position using numpy."""

    def __init__(self, map_shape):
        self.h, self.w = map_shape
        self.grid = np.zeros((self.h, self.w), dtype=np.float64)
        self._initialized = False

    def initialize(self, known_map):
        if self._initialized:
            return
        empty_mask = (known_map != 1)
        count = np.sum(empty_mask)
        if count > 0:
            self.grid[empty_mask] = 1.0 / count
        self._initialized = True

    def observe(self, map_state, enemy_pos, known_map):
        if enemy_pos is not None:
            self.grid[:] = 0.0
            self.grid[enemy_pos[0], enemy_pos[1]] = 1.0
            return

        # Zero out all currently visible empty cells (since enemy is not there)
        self.grid[map_state == 0] = 0.0
        self.grid[known_map == 1] = 0.0

        total = self.grid.sum()
        if total > 1e-12:
            self.grid /= total
        else:
            empty_mask = (known_map != 1)
            count = np.sum(empty_mask)
            if count > 0:
                self.grid[empty_mask] = 1.0 / count

    def predict_movement(self, known_map):
        """Diffuse probability biased toward high mobility/junctions and away from dead-ends."""
        height, width = int(self.h), int(self.w)
        new_grid = np.zeros_like(self.grid)
        
        mobility = _map_data.mobility or {}
        dead_ends = _map_data.dead_ends or set()
        
        for r in range(height):
            for c in range(width):
                prob = self.grid[r, c]
                if prob < 1e-6:
                    continue
                    
                neighbors = []
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width and known_map[nr, nc] != 1:
                        neighbors.append((nr, nc))
                
                if not neighbors:
                    new_grid[r, c] += prob
                    continue
                
                # Assign weights: Stay weight = 0.15 (Ghost moves most of the time)
                stay_weight = 0.15
                weights = []
                for npos in neighbors:
                    weight = 1.0
                    mob = mobility.get(npos, 2)
                    if npos in dead_ends:
                        weight *= 0.1  # Highly avoid dead ends
                    elif mob >= 3:
                        weight *= 1.5  # Prefer junctions
                    weights.append(weight)
                
                total_w = stay_weight + sum(weights)
                new_grid[r, c] += prob * (stay_weight / total_w)
                for npos, w in zip(neighbors, weights):
                    new_grid[npos[0], npos[1]] += prob * (w / total_w)
                    
        new_grid[known_map == 1] = 0.0
        total = new_grid.sum()
        if total > 1e-12:
            new_grid /= total
        self.grid = new_grid

    def get_most_likely_position(self):
        idx = np.argmax(self.grid)
        return (idx // self.w, idx % self.w)

    def get_top_k_positions(self, k=5):
        flat = self.grid.flatten()
        total = flat.sum()
        if total < 1e-12:
            return [((self.h // 2, self.w // 2), 1.0)]
        indices = np.argsort(flat)[::-1][:k]
        positions = []
        for idx in indices:
            prob = flat[idx]
            if prob < 1e-12:
                break
            positions.append(((idx // self.w, idx % self.w), prob))
        if positions:
            total_w = sum(w for _, w in positions)
            if total_w > 1e-12:
                positions = [(p, w / total_w) for p, w in positions]
        return positions if positions else [((self.h // 2, self.w // 2), 1.0)]

    def get_centroid(self):
        total = self.grid.sum()
        if total < 1e-12:
            return (self.h // 2, self.w // 2)
        r_coords = np.arange(self.h).reshape(-1, 1)
        c_coords = np.arange(self.w).reshape(1, -1)
        r_avg = np.sum(self.grid * r_coords) / total
        c_avg = np.sum(self.grid * c_coords) / total
        return (int(round(r_avg)), int(round(c_avg)))


# PacmanAgent (Seek) — v4

class PacmanAgent(BasePacmanAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.name = "Competitive Pacman v4"

        self.known_map = None
        self.belief = None
        self.obs_radius = 5
        self._last_positions = deque(maxlen=12)
        self._map_version = 0
        self._tt = TranspositionTable()
        self._visit_count = {}  # pos -> visit count for anti-loop

    def step(self, map_state, my_position, enemy_position, step_number):
        timer = TimeController(TIME_BUDGET)
        self._last_positions.append(my_position)

        if self.known_map is None:
            self.known_map = np.full_like(map_state, -1)
            self.known_map[map_state == 1] = 1

        prev_sum = self.known_map.sum()
        visible_mask = (map_state != -1)
        self.known_map[visible_mask] = map_state[visible_mask]
        if self.known_map.sum() != prev_sum:
            self._map_version += 1

        _map_data.update(self.known_map, self._map_version)

        if self.belief is None:
            self.belief = BeliefState(map_state.shape)
            self.belief.initialize(self.known_map)

        if step_number > 1:
            self.belief.predict_movement(self.known_map)

        self.belief.observe(map_state, enemy_position, self.known_map)

        nav_map = self.known_map

        # Visit counting for anti-loop
        self._visit_count[my_position] = self._visit_count.get(my_position, 0) + 1

        if enemy_position is not None:
            # Enemy visible: BFS direct pursuit + Minimax
            baseline = self._baseline_pursue(my_position, enemy_position, nav_map)
            best_action = baseline

            if timer.has_time(0.05):
                if step_number % 25 == 1:
                    self._tt.clear()
                best_action = self._minimax_seek(
                    my_position, enemy_position, nav_map, timer, best_action
                )
            return best_action
        else:
            # Enemy NOT visible: Belief-weighted frontier exploration
            return self._belief_frontier_explore(my_position, nav_map, timer)

    def _baseline_pursue(self, my_pos, enemy_pos, nav_map):
        """BFS direct path to enemy — simple and fast."""
        path = bfs_shortest_path(my_pos, enemy_pos, nav_map)
        if path:
            move = path[0]
            steps = max(1, max_valid_steps(my_pos, move, nav_map, self.pacman_speed))
            return (move, steps)
        return self._greedy_toward(my_pos, enemy_pos, nav_map)

    def _baseline_explore(self, my_pos, nav_map):
        """Fallback: unseen or random."""
        unseen_move = bfs_to_nearest_unseen(my_pos, self.known_map)
        if unseen_move is not None:
            steps = max(1, max_valid_steps(my_pos, unseen_move, nav_map, self.pacman_speed))
            return (unseen_move, steps)
        return (Move.STAY, 1)

    def _belief_frontier_explore(self, my_pos, nav_map, timer):
        """Belief-weighted frontier exploration (proven approach).
        Combines: high-belief target pursuit + frontier exploration + anti-loop."""
        belief = self.belief.grid.copy()
        belief[nav_map == 1] = 0.0

        my_dists = bfs_distances(my_pos, nav_map)

        best_belief_target = None
        best_belief_score = -1.0
        h, w = nav_map.shape
        for r in range(h):
            for c in range(w):
                if belief[r, c] < 0.01:
                    continue
                d = my_dists.get((r, c), 999)
                if d == 999:
                    continue
                score = belief[r, c] / (d + 1)
                if score > best_belief_score:
                    best_belief_score = score
                    best_belief_target = (r, c)

        frontier = []
        for r in range(h):
            for c in range(w):
                if nav_map[r, c] == -1:  # Unknown cell
                    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and nav_map[nr, nc] == 0:
                            frontier.append((nr, nc))
                            break

        best_frontier = None
        best_frontier_score = -1.0
        if frontier:
            for f in frontier:
                d = my_dists.get(f, 999)
                if d == 999:
                    continue
                # Local belief around frontier cell
                fr, fc = f
                local_belief = 0.0
                for dr in range(-3, 4):
                    for dc in range(-3, 4):
                        nr, nc = fr + dr, fc + dc
                        if 0 <= nr < h and 0 <= nc < w:
                            local_belief += belief[nr, nc]
                visit_pen = self._visit_count.get(f, 0)
                score = (local_belief * 10 + 1) / (d + 1) - visit_pen * 0.5
                if score > best_frontier_score:
                    best_frontier_score = score
                    best_frontier = f

        target = None
        if best_belief_target is not None and best_belief_score > 0.005:
            target = best_belief_target
        elif best_frontier is not None:
            target = best_frontier
        elif best_belief_target is not None:
            target = best_belief_target
        else:
            # Sweep: go to least-visited reachable cell
            best_sweep_score = float('inf')
            for cell, d in my_dists.items():
                if cell == my_pos:
                    continue
                visits = self._visit_count.get(cell, 0)
                score = visits * 10 + d
                if score < best_sweep_score:
                    best_sweep_score = score
                    target = cell

        if target is None or target == my_pos:
            return self._baseline_explore(my_pos, nav_map)

        path = bfs_shortest_path(my_pos, target, nav_map)
        if path:
            move = path[0]
            steps = max(1, max_valid_steps(my_pos, move, nav_map, self.pacman_speed))
            return (move, steps)

        # Fallback: any valid move, prefer least-visited
        valid = []
        for m in DIR_MOVES:
            npos = apply_move(my_pos, m)
            if is_passable(npos, nav_map):
                valid.append((m, npos))
        if valid:
            valid.sort(key=lambda x: self._visit_count.get(x[1], 0))
            m, _ = valid[0]
            steps = max(1, max_valid_steps(my_pos, m, nav_map, self.pacman_speed))
            return (m, steps)

        return (Move.STAY, 1)

    def _greedy_toward(self, my_pos, target, nav_map):
        best_move = Move.STAY
        best_dist = manhattan(my_pos, target)
        for move in DIR_MOVES:
            npos = apply_move(my_pos, move)
            if is_passable(npos, nav_map):
                d = manhattan(npos, target)
                if d < best_dist:
                    best_dist = d
                    best_move = move
        steps = max(1, max_valid_steps(my_pos, best_move, nav_map, self.pacman_speed))
        return (best_move, steps)


    def _minimax_seek(self, pac_pos, ghost_pos, nav_map, timer, current_best):
        """Iterative deepening Minimax+AB+TT for Pacman (MAX)."""
        best_action = current_best
        dead_ends = _map_data.dead_ends or set()
        mobility = _map_data.mobility or {}
        corridors = _map_data.corridors or set()
        
        # Precompute exact BFS distances from Ghost for wall-aware search
        ghost_dists = bfs_distances(ghost_pos, nav_map)

        last_duration = 0.005
        for depth in range(2, 42, 2):
            iter_start = time.perf_counter()
            # Predict if next depth will exceed the remaining time
            if not timer.has_time(last_duration * 4.0):
                break
            try:
                action, score = self._ab_seek(
                    pac_pos, ghost_pos, nav_map, depth, True,
                    float('-inf'), float('inf'), timer,
                    dead_ends, mobility, corridors, ghost_dists, 0
                )
                if action is not None:
                    best_action = action
                last_duration = time.perf_counter() - iter_start
            except _TimeoutSignal:
                break

        return best_action

    def _ab_seek(self, pac_pos, ghost_pos, nav_map, depth, is_max,
                 alpha, beta, timer, dead_ends, mobility, corridors, ghost_dists, ply):
        """Alpha-Beta+TT: MAX=Pacman (min distance), MIN=Ghost (max distance)."""
        if not timer.has_time(0.005):
            raise _TimeoutSignal()

        dist = ghost_dists.get(pac_pos, manhattan(pac_pos, ghost_pos))
        if dist < 2:
            return None, 10000 - ply  # Caught! Prefer sooner

        # TT Lookup
        tt_result = self._tt.lookup(pac_pos, ghost_pos, depth, alpha, beta, is_max)
        if tt_result is not None:
            return tt_result[1], tt_result[0]

        if depth <= 0:
            return None, self._eval_seek(pac_pos, ghost_pos, dead_ends, mobility, corridors, ghost_dists)

        orig_alpha = alpha
        tt_hint = self._tt.get_best_move(pac_pos, ghost_pos, is_max)

        if is_max:
            best_val = float('-inf')
            best_action = None

            pac_actions = []
            for move in DIR_MOVES:
                ms = max_valid_steps(pac_pos, move, nav_map, self.pacman_speed)
                if ms > 0:
                    new_pos = apply_pacman_move(pac_pos, move, ms, nav_map)
                    pac_actions.append(((move, ms), new_pos))

            # Move ordering: TT hint first, then closest to ghost (exact path dist)
            pac_actions.sort(key=lambda x: ghost_dists.get(x[1], manhattan(x[1], ghost_pos)))
            if tt_hint:
                pac_actions.sort(key=lambda x: (0 if x[0] == tt_hint else 1))

            for action, new_pac_pos in pac_actions:
                _, val = self._ab_seek(
                    new_pac_pos, ghost_pos, nav_map, depth - 1, False,
                    alpha, beta, timer, dead_ends, mobility, corridors, ghost_dists, ply + 1
                )
                if val > best_val:
                    best_val = val
                    best_action = action
                alpha = max(alpha, best_val)
                if alpha >= beta:
                    break

            if best_action is None:
                return (Move.STAY, 1), self._eval_seek(pac_pos, ghost_pos, dead_ends, mobility, corridors, ghost_dists)

            # TT Store
            if best_val <= orig_alpha:
                flag = TT_UPPER
            elif best_val >= beta:
                flag = TT_LOWER
            else:
                flag = TT_EXACT
            self._tt.store(pac_pos, ghost_pos, depth, best_val, flag, best_action, is_max)

            return best_action, best_val
        else:
            best_val = float('inf')
            best_action = None

            ghost_actions = []
            for move in DIR_MOVES:
                npos = apply_move(ghost_pos, move)
                if is_passable(npos, nav_map):
                    ghost_actions.append((move, npos))

            # Move ordering: farthest from pacman first (exact path dist)
            ghost_actions.sort(key=lambda x: -ghost_dists.get(x[1], manhattan(pac_pos, x[1])))
            if tt_hint:
                ghost_actions.sort(key=lambda x: (0 if x[0] == tt_hint else 1))

            for move, new_ghost_pos in ghost_actions:
                _, val = self._ab_seek(
                    pac_pos, new_ghost_pos, nav_map, depth - 1, True,
                    alpha, beta, timer, dead_ends, mobility, corridors, ghost_dists, ply + 1
                )
                if val < best_val:
                    best_val = val
                    best_action = move
                beta = min(beta, best_val)
                if alpha >= beta:
                    break

            if best_action is None:
                return Move.STAY, self._eval_seek(pac_pos, ghost_pos, dead_ends, mobility, corridors, ghost_dists)

            if best_val <= orig_alpha:
                flag = TT_UPPER
            elif best_val >= beta:
                flag = TT_LOWER
            else:
                flag = TT_EXACT
            self._tt.store(pac_pos, ghost_pos, depth, best_val, flag, best_action, is_max)

            return best_action, best_val

    def _eval_seek(self, pac_pos, ghost_pos, dead_ends, mobility, corridors, ghost_dists):
        """O(1) evaluation for Pacman seeking. Higher = better for Pacman."""
        dist = ghost_dists.get(pac_pos, manhattan(pac_pos, ghost_pos))
        effective_dist = math.ceil(dist / self.pacman_speed)

        # Ghost mobility (fewer exits = easier to trap)
        ghost_mob = mobility.get(ghost_pos, 4)
        trap_bonus = (4 - ghost_mob) * 10

        # Dead-end bonus
        de_bonus = 45 if ghost_pos in dead_ends else 0

        # Corridor bonus: Ghost in corridor is easier to trap
        corridor_bonus = 0
        if ghost_pos in corridors and ghost_mob <= 2:
            corridor_bonus = 25
            if effective_dist <= 3:
                corridor_bonus = 60  # Almost trapped!

        # Proximity bonus
        if dist < 2:
            prox = 1000
        elif dist < 4:
            prox = 80
        elif dist < 6:
            prox = 35
        else:
            prox = 0

        return -effective_dist * 12 + trap_bonus + de_bonus + corridor_bonus + prox




# GhostAgent (Hide) — v4

class GhostAgent(BaseGhostAgent):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Competitive Ghost v4"

        self.known_map = None
        self.belief = None
        self.obs_radius = 5
        self._last_positions = deque(maxlen=12)
        self._map_version = 0
        self.max_steps = 200
        self.pacman_speed = 2
        self._last_known_enemy = None
        self._escape_plan = []
        self._escape_target = None
        self._replan_counter = 0
        self._tt = TranspositionTable()

    def step(self, map_state, my_position, enemy_position, step_number):
        timer = TimeController(TIME_BUDGET)
        self._last_positions.append(my_position)

        if self.known_map is None:
            self.known_map = np.full_like(map_state, -1)
            self.known_map[map_state == 1] = 1

        prev_sum = self.known_map.sum()
        visible_mask = (map_state != -1)
        self.known_map[visible_mask] = map_state[visible_mask]
        if self.known_map.sum() != prev_sum:
            self._map_version += 1

        _map_data.update(self.known_map, self._map_version)

        if self.belief is None:
            self.belief = BeliefState(map_state.shape)
            self.belief.initialize(self.known_map)

        if step_number > 1:
            self.belief.predict_movement(self.known_map)

        self.belief.observe(map_state, enemy_position, self.known_map)

        # Track last known Pacman position
        if enemy_position is not None:
            self._last_known_enemy = enemy_position
            self._escape_plan = []  # Cancel plan when we see Pacman

        nav_map = self.known_map

        self._replan_counter += 1
        _following_escape = False

        # Initial escape OR periodic re-planning
        if step_number == 1:
            threat = enemy_position if enemy_position is not None else (my_position[0] + 6, my_position[1])
            self._plan_escape(my_position, nav_map, threat)
        elif enemy_position is None and self._replan_counter >= 3:
            threat = self._last_known_enemy or self.belief.get_most_likely_position()
            self._plan_escape(my_position, nav_map, threat)
            self._replan_counter = 0

        if enemy_position is not None:
            # Enemy visible → Voronoi-lite evasion
            baseline_move = self._voronoi_evasion(my_position, enemy_position, nav_map)
        elif self._escape_plan:
            # Follow escape plan — TRUST IT, don't override with MC rollout
            baseline_move = self._escape_plan.pop(0)
            npos = apply_move(my_position, baseline_move)
            if not is_passable(npos, nav_map):
                self._escape_plan = []
                baseline_move = self._safe_wander(my_position, nav_map, step_number)
            else:
                _following_escape = True
        else:
            # No escape plan — auto-replan with most likely position
            threat = self._last_known_enemy or self.belief.get_most_likely_position()
            self._plan_escape(my_position, nav_map, threat)
            if self._escape_plan:
                baseline_move = self._escape_plan.pop(0)
                npos = apply_move(my_position, baseline_move)
                if is_passable(npos, nav_map):
                    _following_escape = True
                else:
                    self._escape_plan = []
                    baseline_move = self._safe_wander(my_position, nav_map, step_number)
            else:
                baseline_move = self._safe_wander(my_position, nav_map, step_number)

        best_move = baseline_move

        # When following escape plan: use time for VERIFICATION only.
        # Run minimax against believed Pacman position to check if escape
        # direction is dangerously wrong. Only override if significantly better.
        if _following_escape:
            if timer.has_time(0.2):
                # Evaluate escape move vs alternatives
                threat = self._last_known_enemy or self.belief.get_most_likely_position()
                # Quick minimax check
                try:
                    st = TimeController(min(0.3, timer.remaining() - 0.05))
                    alt_move = self._minimax_hide(
                        my_position, threat, nav_map, st, best_move, step_number
                    )
                    # Only override if minimax picks a DIFFERENT move AND
                    # the escape move would go toward Pacman
                    if alt_move != best_move:
                        alt_npos = apply_move(my_position, alt_move)
                        esc_npos = apply_move(my_position, best_move)
                        if is_passable(alt_npos, nav_map) and is_passable(esc_npos, nav_map):
                            threat_dists = bfs_distances(threat, nav_map)
                            alt_d = threat_dists.get(alt_npos, 0)
                            esc_d = threat_dists.get(esc_npos, 0)
                            # Only switch if escape is moving TOWARD Pacman
                            # AND minimax move goes AWAY
                            if esc_d < threat_dists.get(my_position, 0) and alt_d > esc_d:
                                best_move = alt_move
                                self._escape_plan = []  # Cancel bad escape plan
                except _TimeoutSignal:
                    pass
            return best_move

        if not timer.has_time(0.05):
            return best_move

        if step_number % 25 == 1:
            self._tt.clear()

        if enemy_position is not None:
            best_move = self._minimax_hide(
                my_position, enemy_position, nav_map, timer, best_move, step_number
            )
        else:
            best_move = self._mc_rollout_hide(
                my_position, nav_map, timer, best_move, step_number
            )

        return best_move


    def _plan_escape(self, my_pos, nav_map, threat_pos):
        """Plan escape to safe position using weighted random selection."""
        import random
        my_dists = bfs_distances(my_pos, nav_map)
        threat_dists = bfs_distances(threat_pos, nav_map)
        dead_ends = _map_data.dead_ends or set()
        mobility = _map_data.mobility or {}
        corridors = _map_data.corridors or set()
        junctions = _map_data.junctions or set()

        candidates = []

        for pos, my_dist in my_dists.items():
            if my_dist < 1 or my_dist > 15:
                continue
            mob = mobility.get(pos, 0)
            if pos in dead_ends:
                continue

            threat_dist = threat_dists.get(pos, 0)
            threat_eff = math.ceil(threat_dist / self.pacman_speed)

            # Ghost must arrive before Pacman!
            if my_dist >= threat_eff:
                continue

            margin = threat_eff - my_dist

            junction_bonus = 30 if pos in junctions else 0
            corridor_penalty = -25 if pos in corridors else 0
            los_bonus = 0
            if pos[0] != threat_pos[0] and pos[1] != threat_pos[1]:
                los_bonus = 20

            col_diff = abs(pos[1] - threat_pos[1])
            row_diff = abs(pos[0] - threat_pos[0])
            direction_bonus = col_diff + row_diff

            score = (threat_eff * 8
                     + margin * 12
                     + mob * 6
                     + junction_bonus
                     + corridor_penalty
                     + los_bonus
                     + direction_bonus * 2
                     - my_dist * 2)

            candidates.append((pos, score))

        if candidates:
            candidates.sort(key=lambda x: -x[1])
            top_n = min(5, len(candidates))
            top = candidates[:top_n]
            min_s = top[-1][1]
            weights = [max(1, s - min_s + 1) for _, s in top]
            total_w = sum(weights)
            r = random.random() * total_w
            cumul = 0
            chosen = top[0][0]
            for (pos, _), w in zip(top, weights):
                cumul += w
                if r <= cumul:
                    chosen = pos
                    break
            path = safe_shortest_path(my_pos, chosen, nav_map, threat_pos)
            if path:
                self._escape_plan = path[:12]
                self._escape_target = chosen


    def _voronoi_evasion(self, my_pos, pac_pos, nav_map):
        """Pick move maximizing territory control + distance from Pacman."""
        pac_bfs = bfs_distances(pac_pos, nav_map)
        dead_ends = _map_data.dead_ends or set()
        mobility = _map_data.mobility or {}
        corridors = _map_data.corridors or set()

        best_move = Move.STAY
        best_score = float('-inf')

        for move in DIR_MOVES:
            npos = apply_move(my_pos, move)
            if not is_passable(npos, nav_map):
                continue

            # BFS distance from Pacman to this candidate
            pac_dist = pac_bfs.get(npos, 0)
            pac_eff = math.ceil(pac_dist / self.pacman_speed)

            # Territory estimate: BFS from candidate, count cells we control
            npos_bfs = bfs_distances_limited(npos, nav_map, max_dist=10)
            territory = 0
            for cell, g_dist in npos_bfs.items():
                p_dist = pac_bfs.get(cell, 999)
                p_eff = math.ceil(p_dist / self.pacman_speed)
                if g_dist < p_eff:
                    territory += 1

            # Escape routes from this position
            esc_routes = count_escape_routes(npos, pac_bfs, nav_map)

            # Anti-corridor: MASSIVE penalty for entering corridors near Pacman
            de_penalty = 0
            mob = mobility.get(npos, 0)
            if npos in dead_ends:
                de_penalty = -500
            elif npos in corridors and pac_eff < 6:
                de_penalty = -200
            elif npos in corridors:
                de_penalty = -30

            corr_penalty = -20 if mob <= 2 and npos not in corridors else 0

            # Revisit penalty — STRONG anti-oscillation
            rev_penalty = 0
            if len(self._last_positions) >= 1 and npos == self._last_positions[-1]:
                rev_penalty = -100
            elif npos in self._last_positions:
                rev_penalty = -25

            # LOS-breaking bonus
            los_bonus = 0
            if npos[0] != pac_pos[0] and npos[1] != pac_pos[1]:
                los_bonus = 30
            elif npos[0] == pac_pos[0]:
                c_min, c_max = min(npos[1], pac_pos[1]), max(npos[1], pac_pos[1])
                for c in range(c_min + 1, c_max):
                    if nav_map[npos[0], c] == 1:
                        los_bonus = 20
                        break
            elif npos[1] == pac_pos[1]:
                r_min, r_max = min(npos[0], pac_pos[0]), max(npos[0], pac_pos[0])
                for r in range(r_min + 1, r_max):
                    if nav_map[r, npos[1]] == 1:
                        los_bonus = 20
                        break

            score = (pac_eff * 12
                     + territory * 2
                     + esc_routes * 18
                     + de_penalty
                     + corr_penalty
                     + rev_penalty
                     + los_bonus
                     + mob * 5)

            if score > best_score:
                best_score = score
                best_move = move

        return best_move


    def _safe_wander(self, my_pos, nav_map, step_number):
        """Wander safely when Pacman not visible. ZERO random moves."""
        dead_ends = _map_data.dead_ends or set()
        mobility = _map_data.mobility or {}
        corridors = _map_data.corridors or set()
        junctions = _map_data.junctions or set()

        threat_pos = self._last_known_enemy or self.belief.get_most_likely_position()
        threat_dists = bfs_distances(threat_pos, nav_map)

        best_move = Move.STAY
        best_score = float('-inf')
        second_best_move = Move.STAY
        second_best_score = float('-inf')

        for move in DIR_MOVES:
            npos = apply_move(my_pos, move)
            if not is_passable(npos, nav_map):
                continue

            dist_from_threat = threat_dists.get(npos, 0)
            threat_eff = math.ceil(dist_from_threat / self.pacman_speed)
            mob = mobility.get(npos, 0)

            de_penalty = -200 if npos in dead_ends else 0
            corr_penalty = -15 if mob <= 2 else 0

            # STRONG anti-oscillation: massive penalty for returning to last position
            rev_penalty = 0
            if len(self._last_positions) >= 1 and npos == self._last_positions[-1]:
                rev_penalty = -80  # Massive penalty for immediate backtrack
            elif npos in self._last_positions:
                rev_penalty = -20

            # LOS-breaking
            los_bonus = 0
            if npos[0] != threat_pos[0] and npos[1] != threat_pos[1]:
                los_bonus = 25  # Much higher — hiding > running

            # Junction preference
            junction_bonus = 15 if npos in junctions else 0

            score = (threat_eff * 10
                     + mob * 8
                     + de_penalty
                     + corr_penalty
                     + rev_penalty
                     + los_bonus
                     + junction_bonus)

            if score > best_score:
                second_best_score = best_score
                second_best_move = best_move
                best_score = score
                best_move = move
            elif score > second_best_score:
                second_best_score = score
                second_best_move = move

        # Top-2 selection: if scores are within 5%, pick randomly between them
        # This provides unpredictability without self-harm
        if (second_best_score > float('-inf')
                and best_score > 0
                and second_best_score >= best_score * 0.95):
            import random
            if random.random() < 0.3:
                best_move = second_best_move

        return best_move


    def _minimax_hide(self, ghost_pos, pac_pos, nav_map, timer, current_best, step_number):
        """Iterative deepening Minimax+AB+TT for Ghost (MAX)."""
        best_move = current_best
        dead_ends = _map_data.dead_ends or set()
        mobility = _map_data.mobility or {}
        corridors = _map_data.corridors or set()

        last_duration = 0.005
        for depth in range(2, 42, 2):
            iter_start = time.perf_counter()
            # Predict if next depth will exceed the remaining time
            if not timer.has_time(last_duration * 4.0):
                break
            try:
                move, score = self._ab_hide(
                    ghost_pos, pac_pos, nav_map, depth, True,
                    float('-inf'), float('inf'), timer,
                    dead_ends, mobility, corridors, step_number, 0
                )
                if move is not None:
                    best_move = move
                last_duration = time.perf_counter() - iter_start
            except _TimeoutSignal:
                break

        return best_move

    def _ab_hide(self, ghost_pos, pac_pos, nav_map, depth, is_max,
                 alpha, beta, timer, dead_ends, mobility, corridors,
                 step_number, ply):
        """Alpha-Beta+TT: MAX=Ghost (survive), MIN=Pacman (catch)."""
        if not timer.has_time(0.005):
            raise _TimeoutSignal()

        dist = manhattan(ghost_pos, pac_pos)
        if dist < 2:
            return None, -10000 + ply  # Caught — prefer later

        # TT Lookup
        tt_result = self._tt.lookup(ghost_pos, pac_pos, depth, alpha, beta, is_max)
        if tt_result is not None:
            return tt_result[1], tt_result[0]

        if depth <= 0:
            return None, self._eval_hide(ghost_pos, pac_pos, nav_map,
                                         dead_ends, mobility, corridors, step_number)

        orig_alpha = alpha
        tt_hint = self._tt.get_best_move(ghost_pos, pac_pos, is_max)

        if is_max:
            best_val = float('-inf')
            best_action = None

            ghost_actions = []
            for move in DIR_MOVES:
                npos = apply_move(ghost_pos, move)
                if is_passable(npos, nav_map):
                    ghost_actions.append((move, npos))

            # Move ordering: farthest from pacman, TT hint first (exact path dist)
            ghost_actions.sort(key=lambda x: -manhattan(pac_pos, x[1]))
            if tt_hint:
                ghost_actions.sort(key=lambda x: (0 if x[0] == tt_hint else 1))

            for move, new_ghost in ghost_actions:
                _, val = self._ab_hide(
                    new_ghost, pac_pos, nav_map, depth - 1, False,
                    alpha, beta, timer, dead_ends, mobility, corridors,
                    step_number, ply + 1
                )
                if val > best_val:
                    best_val = val
                    best_action = move
                alpha = max(alpha, best_val)
                if alpha >= beta:
                    break

            if best_action is None:
                return Move.STAY, self._eval_hide(ghost_pos, pac_pos, nav_map,
                                                  dead_ends, mobility, corridors, step_number)

            if best_val <= orig_alpha:
                flag = TT_UPPER
            elif best_val >= beta:
                flag = TT_LOWER
            else:
                flag = TT_EXACT
            self._tt.store(ghost_pos, pac_pos, depth, best_val, flag, best_action, is_max)

            return best_action, best_val
        else:
            best_val = float('inf')
            best_action = None

            pac_actions = []
            for move in DIR_MOVES:
                ms = max_valid_steps(pac_pos, move, nav_map, self.pacman_speed)
                if ms > 0:
                    new_pos = apply_pacman_move(pac_pos, move, ms, nav_map)
                    pac_actions.append(((move, ms), new_pos))

            # Move ordering: closest to ghost first (exact path dist)
            pac_actions.sort(key=lambda x: manhattan(x[1], ghost_pos))
            if tt_hint:
                pac_actions.sort(key=lambda x: (0 if x[0] == tt_hint else 1))

            for action, new_pac in pac_actions:
                _, val = self._ab_hide(
                    ghost_pos, new_pac, nav_map, depth - 1, True,
                    alpha, beta, timer, dead_ends, mobility, corridors,
                    step_number, ply + 1
                )
                if val < best_val:
                    best_val = val
                    best_action = action
                beta = min(beta, best_val)
                if alpha >= beta:
                    break

            if best_action is None:
                return (Move.STAY, 1), self._eval_hide(ghost_pos, pac_pos, nav_map,
                                                       dead_ends, mobility, corridors, step_number)

            if best_val <= orig_alpha:
                flag = TT_UPPER
            elif best_val >= beta:
                flag = TT_LOWER
            else:
                flag = TT_EXACT
            self._tt.store(ghost_pos, pac_pos, depth, best_val, flag, best_action, is_max)

            return best_action, best_val

    def _eval_hide(self, ghost_pos, pac_pos, nav_map, dead_ends, mobility,
                   corridors, step_number):
        """O(1) evaluation for Ghost hiding. Higher = better for Ghost."""
        dist = manhattan(ghost_pos, pac_pos)
        effective_dist = math.ceil(dist / self.pacman_speed)

        mob = mobility.get(ghost_pos, 0)

        # Dead-end: catastrophic
        de_penalty = -250 if ghost_pos in dead_ends else 0

        # Corridor: very dangerous
        corr_penalty = 0
        if ghost_pos in corridors and mob <= 2:
            corr_penalty = -40 if effective_dist > 4 else -100

        # Mobility bonus
        mobility_bonus = mob * 10

        # Survival progress bonus
        survival = step_number / self.max_steps

        # Wall-adjacent (LOS breaking)
        wall_adj = 0
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            adj = (ghost_pos[0] + dr, ghost_pos[1] + dc)
            if not is_passable(adj, nav_map):
                wall_adj += 1
        wall_bonus = wall_adj * 3 if wall_adj <= 2 else 0

        # LOS off-axis bonus
        los_bonus = 0
        if ghost_pos[0] != pac_pos[0] and ghost_pos[1] != pac_pos[1]:
            los_bonus = 12

        score = (effective_dist * 12
                 + mobility_bonus
                 + de_penalty
                 + corr_penalty
                 + wall_bonus
                 + los_bonus
                 + survival * 25)

        # Proximity danger penalties
        if dist < 2:
            score -= 500
        elif dist < 4:
            score -= 100
        elif dist < 6:
            score -= 30

        return score


    def _mc_rollout_hide(self, ghost_pos, nav_map, timer, current_best, step_number):
        """MC rollouts when Pacman not visible."""
        candidates = self.belief.get_top_k_positions(k=15)
        if not candidates:
            return current_best

        dead_ends = _map_data.dead_ends or set()
        mobility = _map_data.mobility or {}

        move_scores = {}
        valid_moves = []
        for move in DIR_MOVES:
            npos = apply_move(ghost_pos, move)
            if is_passable(npos, nav_map):
                # Anti-dead-end in rollouts too
                if npos not in dead_ends:
                    valid_moves.append((move, npos))
        if not valid_moves:
            # Fallback: include all valid
            for move in DIR_MOVES:
                npos = apply_move(ghost_pos, move)
                if is_passable(npos, nav_map):
                    valid_moves.append((move, npos))
        if not valid_moves:
            valid_moves = [(Move.STAY, ghost_pos)]

        for move, new_ghost in valid_moves:
            if not timer.has_time(0.03):
                break
            total_score = 0.0
            for pac_pos, weight in candidates:
                if not timer.has_time(0.015):
                    break
                surv = self._rollout(new_ghost, pac_pos, nav_map, 50, timer)
                total_score += surv * weight
            move_scores[move] = total_score

        if move_scores:
            return max(move_scores, key=move_scores.get)
        return current_best

    def _rollout(self, ghost_pos, pac_pos, nav_map, max_steps, timer):
        """Simulate: Ghost smart evasion, Pacman greedy+speed2."""
        g = ghost_pos
        p = pac_pos
        dead_ends = _map_data.dead_ends or set()

        for step in range(max_steps):
            if timer and not timer.has_time(0.004):
                return step
            if manhattan(g, p) < 2:
                return step

            # Ghost: maximize distance, avoid dead ends
            g_nbrs = get_neighbors(g, nav_map)
            if g_nbrs:
                safe = [(pos, m) for pos, m in g_nbrs if pos not in dead_ends]
                if not safe:
                    safe = g_nbrs
                g = max(safe, key=lambda x: manhattan(x[0], p))[0]

            # Pacman: greedy minimize distance + speed 2
            p_nbrs = get_neighbors(p, nav_map)
            if p_nbrs:
                best = min(p_nbrs, key=lambda x: manhattan(x[0], g))
                p = best[0]
                second = apply_move(p, best[1])
                if is_passable(second, nav_map):
                    p = second

        return max_steps
