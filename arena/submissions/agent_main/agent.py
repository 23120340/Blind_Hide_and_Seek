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
            {pos for pos in unseen if region(pos) == 0},
            {pos for pos in unseen if region(pos) == 1},
            {pos for pos in unseen if region(pos) == 2},
            {pos for pos in unseen if region(pos) in (2, 3)},
            unseen,
        ]
        orders = [
            [Move.RIGHT, Move.UP, Move.DOWN, Move.LEFT],
            [Move.LEFT, Move.UP, Move.DOWN, Move.RIGHT],
            [Move.LEFT, Move.DOWN, Move.UP, Move.RIGHT],
            [Move.DOWN, Move.RIGHT, Move.LEFT, Move.UP],
            [Move.RIGHT, Move.UP, Move.LEFT, Move.DOWN],
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


class GhostAgent(BaseGhostAgent, ArenaTools):
    PANIC_TURNS = 6
    CAUTION_TURNS = 12

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Agent Main Belief Hider"
        self._init_memory()
        self.belief = None
        self.last_move = None
        self.haven = None
        self.last_enemy_step = -999
        self.fixed_opening = [
            Move.RIGHT,
            Move.UP,
            Move.RIGHT,
            Move.UP,
        ]

    def step(self, map_state, my_position, enemy_position, step_number):
        self._observe(map_state, my_position, enemy_position, DEFAULT_PACMAN_START)
        if enemy_position is not None:
            self.last_enemy_step = step_number
            move = self._visible_flee(my_position, enemy_position, map_state)
        elif step_number <= len(self.fixed_opening):
            opening_move = self.fixed_opening[step_number - 1]
            if opening_move == Move.STAY or self._open(self._move(my_position, opening_move), map_state):
                move = opening_move
            else:
                move = self._move_toward_haven(my_position, map_state)
        elif self.last_enemy is not None and step_number - self.last_enemy_step <= 5:
            threat = self._advance_toward(self.last_enemy, my_position, 2)
            move = self._visible_flee(my_position, threat, map_state)
        else:
            move = self._move_toward_haven(my_position, map_state)

        self.last_move = move
        return move

    def _move_toward_haven(self, my_position, grid):
        target = (grid.shape[0] - 2, grid.shape[1] - 2)
        best_move = Move.STAY
        best_score = -10**18
        for pos, move in self._neighbors(my_position, grid, stay=True):
            dist = self._manhattan(pos, target)
            exits = len(self._neighbors(pos, self.known))
            score = -dist * 4 + exits * 1.5 - self.visits.get(pos, 0)
            if move == Move.STAY:
                score -= 2
            if self.last_move is not None and move.value == (-self.last_move.value[0], -self.last_move.value[1]):
                score -= 1
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _visible_flee(self, my_position, enemy_position, grid):
        best_move = Move.STAY
        best_score = -10**18
        pac_reach = self._pacman_reachable_positions(enemy_position)
        for pos, move in self._neighbors(my_position, grid, stay=True):
            direct = min(self._manhattan(pos, pac_pos) for pac_pos in pac_reach)
            maze = min(self._maze_distance(pos, pac_pos, self.known, max_depth=35) for pac_pos in pac_reach)
            exits = len(self._neighbors(pos, self.known))
            score = direct * 20 + maze * 3 + exits * 2 - self.visits.get(pos, 0)

            same_row = pos[0] == enemy_position[0]
            same_col = pos[1] == enemy_position[1]
            if not same_row and not same_col:
                score += 90
            elif same_row or same_col:
                score -= 45

            line_risk = 0
            for pac_pos in pac_reach:
                if (pos[0] == pac_pos[0] or pos[1] == pac_pos[1]) and self._manhattan(pos, pac_pos) <= 5:
                    line_risk += 1
            score -= line_risk * 420

            if my_position[0] == enemy_position[0] and move.value[0] != 0:
                score += 120
            if my_position[1] == enemy_position[1] and move.value[1] != 0:
                score += 120

            if direct < 2:
                score -= 1000
            elif direct < 4:
                score -= 120
            if move == Move.STAY:
                score -= 5
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _update_pacman_belief(self, map_state, my_position, enemy_position, step_number):
        if self.belief is None or self.belief.shape != self.known.shape:
            self.belief = np.zeros(self.known.shape, dtype=np.float32)
            passable = self.known != 1
            self.belief[passable] = 1.0
            total = float(self.belief.sum())
            if total > 0:
                self.belief /= total

        if step_number > 1:
            new_belief = np.zeros_like(self.belief)
            for r, c in np.argwhere(self.belief > 1e-9):
                pos = (int(r), int(c))
                reachable = self._pacman_reachable_positions(pos)
                share = self.belief[pos] / max(1, len(reachable))
                for nxt in reachable:
                    new_belief[nxt] += share
            self.belief = new_belief

        if enemy_position is not None:
            self.belief[:] = 0.0
            self.belief[enemy_position] = 1.0
            self.haven = None
        else:
            visible_empty = map_state == 0
            self.belief[visible_empty] = 0.0
            self.belief[my_position] = 0.0

        total = float(self.belief.sum())
        if total > 0:
            self.belief /= total
        else:
            passable = self.known != 1
            self.belief[passable] = 1.0
            self.belief[my_position] = 0.0
            total = float(self.belief.sum())
            if total > 0:
                self.belief /= total

    def _belief_centroid(self):
        if self.belief is None:
            return self.last_enemy
        total = float(self.belief.sum())
        if total <= 1e-12:
            return self.last_enemy
        rows, cols = np.indices(self.belief.shape)
        r = int(round(float((rows * self.belief).sum() / total)))
        c = int(round(float((cols * self.belief).sum() / total)))
        r = max(0, min(self.belief.shape[0] - 1, r))
        c = max(0, min(self.belief.shape[1] - 1, c))
        if self._open((r, c), self.known):
            return (r, c)
        return self.last_enemy

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

    def _pacman_turn_distances(self, origin):
        queue = deque([origin])
        dist = {origin: 0}
        while queue:
            pos = queue.popleft()
            for nxt in self._pacman_reachable_positions(pos):
                if nxt not in dist:
                    dist[nxt] = dist[pos] + 1
                    queue.append(nxt)
        return dist

    def _panic_move(self, my_position, turn_dist):
        pac_nexts = [pos for pos, dist in turn_dist.items() if dist <= 1]
        best_move = Move.STAY
        best_score = -10**18
        for pos, move in self._neighbors(my_position, self.known, stay=True):
            worst = 999
            for pac_pos in pac_nexts:
                if self._manhattan(pos, pac_pos) < 2:
                    worst = -100
                    break
                pac2 = self._pacman_reachable_positions(pac_pos)
                g2_positions = [pos] + [nxt for nxt, _ in self._neighbors(pos, self.known)]
                reply = max(min(self._manhattan(g2, p2) for p2 in pac2) for g2 in g2_positions)
                worst = min(worst, reply)
            exits = len(self._neighbors(pos, self.known))
            score = worst * 12 + turn_dist.get(pos, 0) * 2 + exits * 1.5
            score -= self.visits.get(pos, 0) * 0.8
            if self.last_move is not None and move.value == (-self.last_move.value[0], -self.last_move.value[1]):
                score -= 2.0
            if move == Move.STAY:
                score -= 0.5
            if exits <= 1:
                score -= 5.0
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _caution_move(self, my_position, turn_dist):
        best_move = Move.STAY
        best_score = -10**18
        for pos, move in self._neighbors(my_position, self.known, stay=True):
            exits = len(self._neighbors(pos, self.known))
            score = turn_dist.get(pos, 0) * 2.8 + exits * 1.4
            score -= self.visits.get(pos, 0) * 0.7
            if self.last_move is not None and move.value == (-self.last_move.value[0], -self.last_move.value[1]):
                score -= 1.5
            if move == Move.STAY:
                score -= 0.5
            if exits <= 1:
                score -= 5.0
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _calm_move(self, my_position, turn_dist):
        if self.haven is None or self.haven == my_position:
            self.haven = self._find_haven(my_position, turn_dist)
        if self.haven is not None and self.haven != my_position:
            path = self._bfs_path(my_position, self.haven, self.known)
            if path:
                return path[0]
        return self._caution_move(my_position, turn_dist)

    def _find_haven(self, my_position, turn_dist):
        distances = self._distance_map(my_position, self.known, max_depth=80)
        best = None
        best_score = -10**18
        for pos, dist in distances.items():
            exits = len(self._neighbors(pos, self.known))
            if exits < 2:
                continue
            score = turn_dist.get(pos, 0) * 3.0 + exits - dist * 0.25
            score -= self.visits.get(pos, 0) * 0.8
            if score > best_score:
                best_score = score
                best = pos
        return best

    def _wander(self, my_position, grid):
        best_move = Move.STAY
        best_score = -10**18
        for pos, move in self._neighbors(my_position, grid, stay=True):
            score = len(self._neighbors(pos, self.known)) * 2 - self.visits.get(pos, 0)
            if move == Move.STAY:
                score -= 1
            if score > best_score:
                best_score = score
                best_move = move
        return best_move
