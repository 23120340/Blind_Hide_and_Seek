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
        self._init_memory()

    def step(self, map_state, my_position, enemy_position, step_number):
        self._observe(map_state, my_position, enemy_position)

        if enemy_position is not None:
            target = self._trap_target(enemy_position)
        else:
            target = self._search_target(my_position)

        if target is not None:
            path = self._bfs_path(my_position, target, self.known)
            if path:
                return self._path_action(my_position, path, map_state)

        return self._fallback(my_position, map_state)

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
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Agent Main Escape Hider"
        self._init_memory()
        self.fixed_opening = [
            Move.RIGHT,
            Move.RIGHT,
            Move.RIGHT,
            Move.UP,
            Move.UP,
            Move.LEFT,
        ]

    def step(self, map_state, my_position, enemy_position, step_number):
        self._observe(map_state, my_position, enemy_position, DEFAULT_PACMAN_START)
        threat = enemy_position or self.last_enemy

        if self.fixed_map_mode and step_number <= len(self.fixed_opening):
            opening_move = self.fixed_opening[step_number - 1]
            opening_pos = self._move(my_position, opening_move)
            if self._open(opening_pos, map_state):
                return opening_move

        # Refresh articulation cache (cheap when the known-map is unchanged).
        self._compute_articulation()

        best_move = Move.STAY
        best_score = -10**18
        for nxt, move in self._neighbors(my_position, map_state, stay=True):
            score = self._escape_score(nxt, move, threat)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _escape_score(self, pos, move, threat):
        exits = len(self._neighbors(pos, self.known))
        potential = self._escape_potential(pos, threat)
        cycle = self._biconn_size.get(pos, 0) if self._biconn_size else 0
        is_art = self._art_points and pos in self._art_points

        score = exits * 10
        score += potential * 2.5
        score += cycle * 0.8                          # bigger cycle = safer to loop
        score += self._unknown_adjacent(pos) * 1.5
        score -= self.visits.get(pos, 0) * 2.5

        if is_art:
            score -= 35                               # articulation = chokepoint
        if pos in list(self.recent)[-4:]:
            score -= 8
        if exits <= 1:
            score -= 85
        elif exits == 2 and self._corridor_without_junction(pos):
            score -= 22
        if move == Move.STAY:
            score -= 12

        if threat is not None:
            pacman_reach = self._pacman_reachable_positions(threat)
            min_direct = min(self._manhattan(pos, pac_pos) for pac_pos in pacman_reach)
            min_maze = min(self._maze_distance(pos, pac_pos, self.known, max_depth=32) for pac_pos in pacman_reach)

            score += min(self._maze_distance(pos, threat, self.known, max_depth=35), 35) * 4.5
            score += min(min_maze, 32) * 5
            score += min_direct * 8

            # Voronoi-style bonus: more ghost-first territory = more room.
            if min_direct >= 2:
                ghost_terr = self._voronoi_terr(threat, pos, pacman_speed=2)
                score += ghost_terr * 0.4

            if min_direct < 2:
                score -= 700
            elif min_direct < 4:
                score -= 90

        return score

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

    def _escape_potential(self, start, threat):
        distances = self._distance_map(start, self.known, max_depth=7)
        best = 0
        for pos, dist in distances.items():
            exits = len(self._neighbors(pos, self.known))
            score = exits * 3 - dist * 0.5 - self.visits.get(pos, 0)
            if threat is not None:
                score += min(self._maze_distance(pos, threat, self.known, max_depth=25), 25)
            best = max(best, score)
        return best

    def _corridor_without_junction(self, start):
        queue = deque([(start, 0)])
        seen = {start}
        while queue:
            pos, dist = queue.popleft()
            if dist > 5:
                continue
            if len(self._neighbors(pos, self.known)) >= 3:
                return False
            for nxt, _ in self._neighbors(pos, self.known):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append((nxt, dist + 1))
        return True
