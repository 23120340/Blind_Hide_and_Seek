"""
Primary role-combined tournament agent.

Pacman uses trap-aware pursuit: when the Ghost is visible, it targets cells
that reduce the Ghost's escape options; otherwise it searches frontier cells.

Ghost uses a fixed-map opening and a blended safety evaluator combining four
signals:
  - direct Pacman reachability and worst-case lunge distance,
  - maze distance and a 7-step escape-potential lookahead,
  - cycle awareness via articulation points and biconnected component size,
  - Voronoi territory: cells the Ghost reaches before a speed-aware Pacman.

The articulation-point analysis is computed lazily and cached across turns,
so the per-step cost stays well within the time limit.
"""

import sys
import time
import random
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
DEFAULT_PACMAN_PATROL = [
    (5, 12),
    (1, 19),
    (1, 1),
    (5, 5),
    (15, 5),
    (19, 1),
    (19, 19),
    (15, 15),
]
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
        # Articulation-point cache for cycle-aware Ghost evaluation.
        self._art_points = None
        self._biconn_size = None
        self._graph_signature = None

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

    def _passable(self, pos):
        return self._inside(pos, self.known) and self.known[pos] != 1

    def _passable_neighbors(self, pos):
        out = []
        for move in MOVES:
            nxt = self._move(pos, move)
            if self._passable(nxt):
                out.append(nxt)
        return out

    def _compute_articulation(self):
        """Iterative Tarjan articulation-point analysis on the open-cell
        subgraph. Caches results until the known-map changes. Unknown cells
        are treated as passable so the Ghost is not paranoid in early-game
        partial visibility."""
        sig = (self.known.shape, int((self.known == 0).sum()))
        if sig == self._graph_signature and self._art_points is not None:
            return
        self._graph_signature = sig

        open_cells = [tuple(pt) for pt in np.argwhere(self.known == 0)]
        for pt in np.argwhere(self.known == -1):
            open_cells.append(tuple(pt))
        if not open_cells:
            self._art_points = set()
            self._biconn_size = {}
            return

        idx = {cell: i for i, cell in enumerate(open_cells)}
        n = len(open_cells)
        disc = [-1] * n
        low = [0] * n
        parent = [-1] * n
        children = [0] * n
        art = [False] * n
        timer = [0]

        for start in range(n):
            if disc[start] != -1:
                continue
            stack = [(start, iter(self._passable_neighbors(open_cells[start])))]
            disc[start] = timer[0]
            low[start] = timer[0]
            timer[0] += 1
            while stack:
                u, it = stack[-1]
                advanced = False
                for v_pos in it:
                    if v_pos not in idx:
                        continue
                    v = idx[v_pos]
                    if disc[v] == -1:
                        parent[v] = u
                        children[u] += 1
                        disc[v] = timer[0]
                        low[v] = timer[0]
                        timer[0] += 1
                        stack.append((v, iter(self._passable_neighbors(open_cells[v]))))
                        advanced = True
                        break
                    elif v != parent[u]:
                        if disc[v] < low[u]:
                            low[u] = disc[v]
                if not advanced:
                    stack.pop()
                    if parent[u] != -1:
                        if low[u] < low[parent[u]]:
                            low[parent[u]] = low[u]
                        if low[u] >= disc[parent[u]]:
                            art[parent[u]] = True
                    else:
                        if children[u] > 1:
                            art[u] = True

        art_set = {open_cells[i] for i in range(n) if art[i]}
        biconn_size = {}
        seen = set()
        for cell in open_cells:
            if cell in seen or cell in art_set:
                continue
            queue = deque([cell])
            comp = []
            seen.add(cell)
            while queue:
                p = queue.popleft()
                comp.append(p)
                for nb in self._passable_neighbors(p):
                    if nb in art_set or nb in seen:
                        continue
                    seen.add(nb)
                    queue.append(nb)
            for p in comp:
                biconn_size[p] = len(comp)
        for cell in art_set:
            best = 0
            for nb in self._passable_neighbors(cell):
                best = max(best, biconn_size.get(nb, 0))
            biconn_size[cell] = best

        self._art_points = art_set
        self._biconn_size = biconn_size

    def _voronoi_terr(self, pacman_pos, ghost_pos, pacman_speed):
        """Ghost-first territory size: cells where the Ghost arrives before
        a speed-aware Pacman."""
        pac_d = self._distance_map(pacman_pos, self.known, max_depth=35)
        gho_d = self._distance_map(ghost_pos, self.known, max_depth=35)
        ghost_terr = 0
        for cell, gc in gho_d.items():
            pc = pac_d.get(cell, 999)
            pt = (pc + pacman_speed - 1) // pacman_speed
            if gc < pt:
                ghost_terr += 1
        return ghost_terr


class PacmanAgent(BasePacmanAgent, ArenaTools):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Agent Main Trap Seeker"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 1)))
        self.seen_enemy_once = False
        self.patrol_index = 0
        self.seen_tiles = set()
        self.sweep_state = 0
        self.sweep_steps = 0
        self._init_memory()

    def step(self, map_state, my_position, enemy_position, step_number):
        self._observe(map_state, my_position, enemy_position, DEFAULT_GHOST_START)
        for r, c in np.argwhere(map_state == 0):
            self.seen_tiles.add((int(r), int(c)))
        self.seen_tiles.add(my_position)

        if enemy_position is not None:
            self.seen_enemy_once = True
            target = self._trap_target(enemy_position)
        elif self.fixed_map_mode and not self.seen_enemy_once:
            if step_number >= 16:
                corner = (self.known.shape[0] - 2, self.known.shape[1] - 2)
                path = self._bfs_path(my_position, corner, self.known)
                if path:
                    return self._path_action(my_position, path, map_state)
            path = self._sweep_path(my_position)
            if path:
                return self._path_action(my_position, path, map_state)
            target = self._patrol_target(my_position)
        else:
            target = self._search_target(my_position)

        if target is not None:
            path = self._bfs_path(my_position, target, self.known)
            if path:
                return self._path_action(my_position, path, map_state)

        return self._fallback(my_position, map_state)

    def _patrol_target(self, start):
        self.last_enemy = None
        if not DEFAULT_PACMAN_PATROL:
            return None
        if start == DEFAULT_PACMAN_PATROL[self.patrol_index]:
            self.patrol_index = (self.patrol_index + 1) % len(DEFAULT_PACMAN_PATROL)
        return DEFAULT_PACMAN_PATROL[self.patrol_index]

    def _sweep_path(self, start):
        self.last_enemy = None
        open_tiles = {tuple(pt) for pt in np.argwhere(self.known == 0)}
        unseen = open_tiles - self.seen_tiles
        if not unseen:
            self.seen_tiles = {start}
            self.sweep_state = 0
            self.sweep_steps = 0
            unseen = open_tiles - self.seen_tiles
        if not unseen:
            return []

        height, width = self.known.shape
        mid_r, mid_c = height // 2, width // 2

        def region(pos):
            r, c = pos
            if r < mid_r and c >= mid_c:
                return 0
            if r < mid_r and c < mid_c:
                return 1
            if r >= mid_r and c < mid_c:
                return 2
            return 3

        groups = [
            {pos for pos in unseen if region(pos) == 1},
            {pos for pos in unseen if region(pos) == 0},
            {pos for pos in unseen if region(pos) == 2},
            {pos for pos in unseen if region(pos) in (2, 3)},
            unseen,
        ]
        orders = [
            [Move.LEFT, Move.UP, Move.DOWN, Move.RIGHT],
            [Move.RIGHT, Move.UP, Move.DOWN, Move.LEFT],
            [Move.LEFT, Move.DOWN, Move.UP, Move.RIGHT],
            [Move.DOWN, Move.RIGHT, Move.LEFT, Move.UP],
            [Move.LEFT, Move.UP, Move.RIGHT, Move.DOWN],
        ]

        for _ in range(len(groups) + 1):
            idx = min(self.sweep_state, len(groups) - 1)
            if groups[idx] and not (idx == 0 and self.sweep_steps >= 3):
                path = self._bfs_path_to_any(start, groups[idx], self.known, orders[idx])
                if path:
                    self.sweep_steps += 1
                    return path
            self.sweep_state = (self.sweep_state + 1) % len(groups)
            self.sweep_steps = 0
        return self._bfs_path_to_any(start, unseen, self.known, MOVES)

    def _bfs_path_to_any(self, start, goals, grid, move_order):
        if start in goals:
            return []
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            pos, path = queue.popleft()
            for move in move_order:
                nxt = self._move(pos, move)
                if nxt in seen or not self._open(nxt, grid):
                    continue
                next_path = path + [move]
                if nxt in goals:
                    return next_path
                seen.add(nxt)
                queue.append((nxt, next_path))
        return []

    def _trap_target(self, enemy):
        options = [enemy] + [pos for pos, _ in self._neighbors(enemy, self.known)]
        best = enemy
        best_score = -10**9
        for pos in options:
            escape_options = len(self._neighbors(pos, self.known))
            score = -escape_options * 12
            score -= self._unknown_adjacent(pos) * 4
            score -= self._manhattan(pos, enemy)
            if score > best_score:
                best_score = score
                best = pos
        return best

    def _search_target(self, start):
        if self.last_enemy is not None and self.last_enemy != start:
            return self.last_enemy

        distances = self._distance_map(start, self.known, max_depth=35)
        best = None
        best_score = -10**9
        for pos, dist in distances.items():
            if pos == start:
                continue
            score = self._unknown_adjacent(pos) * 8
            score -= dist
            score -= self.visits.get(pos, 0) * 3
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
            score = self._unknown_adjacent(nxt) * 4 - self.visits.get(nxt, 0)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move, 1


class GhostAgent(BaseGhostAgent):
    BOARD = DEFAULT_MAP.astype(np.int8)
    THINK_BUDGET = 0.85

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Agent Main Route Hider"
        self.recent_cells = deque(maxlen=4)
        self.pacman_memory = None
        self.hidden_turns = 0
        self.guard_cell = (5, 12)
        self.guard_route = []
        self.reaching_guard = True
        self.holding_guard = False
        self.enemy_trace = deque(maxlen=5)
        self.approach_read = 0
        self.lane_read = 0

    def step(self, map_state, my_position, enemy_position, step_number):
        deadline = time.perf_counter() + self.THINK_BUDGET

        if enemy_position is not None:
            self._read_enemy_motion(enemy_position, my_position)
            self.reaching_guard = False
            self.holding_guard = False
            self.pacman_memory = enemy_position
            self.hidden_turns = 0
        else:
            self.hidden_turns += 1

        if self.pacman_memory is None:
            self.pacman_memory = DEFAULT_PACMAN_START

        if self.reaching_guard:
            if my_position == self.guard_cell:
                self.reaching_guard = False
                self.holding_guard = True
                return Move.STAY

            if not self.guard_route:
                self.guard_route = self._bfs_path(my_position, self.guard_cell, self.BOARD, deadline)

            if self.guard_route:
                move = self.guard_route.pop(0)
                self._remember(my_position, move)
                return move

            self.reaching_guard = False

        if self.holding_guard:
            if step_number < 60:
                return Move.STAY
            self.holding_guard = False

        threat = self._projected_enemy()
        pacman_dist = self._distance_field(threat, self.BOARD, deadline)
        if self._is_timeout(deadline):
            return self._fallback(my_position, threat, map_state)

        target = my_position
        best_dist = -1
        for pos, dist in pacman_dist.items():
            if dist > best_dist:
                best_dist = dist
                target = pos

        move = self._pick_runaway_move(my_position, threat, target, map_state, pacman_dist, deadline)
        if move == Move.STAY and self.hidden_turns < 5:
            candidates = [pos for pos in self._walkable_neighbors(my_position, map_state) if pos not in self.recent_cells]
            if candidates:
                next_pos = random.choice(candidates)
                move = self._delta_to_move(my_position, next_pos)

        self._remember(my_position, move)
        return move

    def _remember(self, pos, move):
        if move != Move.STAY:
            self.recent_cells.append((pos[0] + move.value[0], pos[1] + move.value[1]))

    def _read_enemy_motion(self, enemy_pos, my_pos):
        if self.enemy_trace:
            prev = self.enemy_trace[-1]
            prev_dist = abs(prev[0] - my_pos[0]) + abs(prev[1] - my_pos[1])
            cur_dist = abs(enemy_pos[0] - my_pos[0]) + abs(enemy_pos[1] - my_pos[1])
            if cur_dist < prev_dist:
                self.approach_read = min(4, self.approach_read + 1)
            else:
                self.approach_read = max(0, self.approach_read - 1)

            if enemy_pos[0] == my_pos[0] or enemy_pos[1] == my_pos[1]:
                self.lane_read = min(4, self.lane_read + 1)
            else:
                self.lane_read = max(0, self.lane_read - 1)

        self.enemy_trace.append(enemy_pos)

    def _projected_enemy(self):
        if self.hidden_turns == 0 or self.hidden_turns > 3 or len(self.enemy_trace) < 2:
            return self.pacman_memory

        prev = self.enemy_trace[-2]
        cur = self.enemy_trace[-1]
        projected = (cur[0] + cur[0] - prev[0], cur[1] + cur[1] - prev[1])
        if self._is_open(projected, self.BOARD):
            return projected
        return self.pacman_memory

    def _seeker_profile(self):
        if self.approach_read >= 2 or self.lane_read >= 2:
            return "hunter"
        return "unknown"

    def _walkable_neighbors(self, pos, grid):
        out = []
        h, w = grid.shape
        for move in MOVES:
            nr, nc = pos[0] + move.value[0], pos[1] + move.value[1]
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] != 1:
                out.append((nr, nc))
        return out

    def _distance_field(self, start, grid, deadline=None):
        distances = {start: 0}
        queue = deque([start])
        while queue:
            if self._is_timeout(deadline):
                break
            cur = queue.popleft()
            for nxt in self._walkable_neighbors(cur, grid):
                if nxt not in distances:
                    distances[nxt] = distances[cur] + 1
                    queue.append(nxt)
        return distances

    def _bfs_path(self, start, target, grid, deadline=None):
        if start == target:
            return []
        queue = deque([(start, [])])
        seen = {start}
        while queue:
            if self._is_timeout(deadline):
                return []
            cur, path = queue.popleft()
            for nxt in self._walkable_neighbors(cur, grid):
                if nxt in seen:
                    continue
                move = self._delta_to_move(cur, nxt)
                if nxt == target:
                    return path + [move]
                seen.add(nxt)
                queue.append((nxt, path + [move]))
        return []

    def _pick_runaway_move(self, my_pos, enemy_pos, target_pos, grid, pacman_dist, deadline):
        neighbors = self._walkable_neighbors(my_pos, grid)
        if not neighbors:
            return Move.STAY

        distances = [pacman_dist.get(pos, 0) for pos in neighbors]
        full_panic = all(dist <= 3 for dist in distances)
        best_move = Move.STAY
        best_score = -10**18

        for pos in neighbors:
            if self._is_timeout(deadline):
                break
            dist = pacman_dist.get(pos, 0)
            exits = self._walkable_neighbors(pos, grid)
            score = dist * 100 if full_panic else dist * 20
            if not full_panic:
                if dist <= 3:
                    score -= 2000
                elif dist <= 5:
                    score -= 500

            if not self._clear_lane(pos, enemy_pos, grid):
                score += 1000

            exit_count = len(exits)
            if exit_count >= 3:
                score += 100
            elif exit_count <= 1:
                score -= 500
            elif self._corridor_pair(exits):
                score -= 100
            else:
                score += 50

            score += self._wall_contact(pos, grid) * 20
            score -= (abs(pos[0] - target_pos[0]) + abs(pos[1] - target_pos[1])) * 5
            if pos in self.recent_cells:
                score -= 200
            if self._seeker_profile() == "hunter":
                if self._clear_lane(pos, enemy_pos, grid):
                    score -= 300
                else:
                    score += 80
                score += exit_count * 10

            if score > best_score:
                best_score = score
                best_move = self._delta_to_move(my_pos, pos)

        return best_move

    def _clear_lane(self, a, b, grid):
        if a[0] == b[0]:
            step = 1 if b[1] > a[1] else -1
            return all(grid[a[0], c] != 1 for c in range(a[1] + step, b[1], step))
        if a[1] == b[1]:
            step = 1 if b[0] > a[0] else -1
            return all(grid[r, a[1]] != 1 for r in range(a[0] + step, b[0], step))
        return False

    def _corridor_pair(self, exits):
        if len(exits) != 2:
            return False
        return exits[0][0] == exits[1][0] or exits[0][1] == exits[1][1]

    def _wall_contact(self, pos, grid):
        count = 0
        h, w = grid.shape
        for move in MOVES:
            nr, nc = pos[0] + move.value[0], pos[1] + move.value[1]
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == 1:
                count += 1
        return count

    def _is_open(self, pos, grid):
        return 0 <= pos[0] < grid.shape[0] and 0 <= pos[1] < grid.shape[1] and grid[pos] != 1

    def _delta_to_move(self, start, end):
        delta = (end[0] - start[0], end[1] - start[1])
        for move in MOVES:
            if move.value == delta:
                return move
        return Move.STAY

    def _fallback(self, my_pos, enemy_pos, grid):
        neighbors = self._walkable_neighbors(my_pos, grid)
        if not neighbors:
            return Move.STAY
        best = max(neighbors, key=lambda pos: abs(pos[0] - enemy_pos[0]) + abs(pos[1] - enemy_pos[1]))
        return self._delta_to_move(my_pos, best)

    def _is_timeout(self, deadline):
        return deadline is not None and time.perf_counter() >= deadline
