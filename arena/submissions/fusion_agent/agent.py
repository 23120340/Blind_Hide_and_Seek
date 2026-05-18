"""
Fusion candidate: best Pacman + best Ghost from current agent pool.

- Pacman role: `Main Frontier Hunter` (the proven fixed_voronoi_agent Pacman) - has the
  fastest average capture in deterministic round-robin.
- Ghost role: `Cycle Dancer` (cycle_agent's Ghost) - uses articulation point
  analysis to stay in big cycle regions; survives strong seekers far longer
  than the fixed_voronoi_agent Ghost in head-to-head testing.

Sharing this agent module gives us a single binary that combines both
strengths so the tournament evaluates the upgrade fairly. If this beats
fixed_voronoi_agent in head-to-head and round-robin, we will fold the Ghost role
into the primary candidate.

Both halves use the same fixed-map prior so the deterministic-start
opening exploit works as in fixed_voronoi_agent.
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
    for r, row in enumerate(DEFAULT_LAYOUT):
        for c, ch in enumerate(row):
            grid[r, c] = 1 if ch == "#" else 0
    return grid


DEFAULT_MAP = _default_map()


class SharedTools:
    def _init_state(self):
        self.known = None
        self.visits = {}
        self.last_enemy = None
        self.recent = deque(maxlen=10)
        self.fixed_map_mode = False
        self._art_points = None
        self._biconn_size = None
        self._graph_signature = None

    def _maybe_enable_fixed_map(self, grid, pos, assumed_enemy):
        if self.fixed_map_mode:
            return
        if grid.shape != DEFAULT_MAP.shape:
            return
        if not np.array_equal(grid == 1, DEFAULT_MAP == 1):
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

    def _passable(self, pos):
        return self._inside(pos, self.known) and self.known[pos] != 1

    def _move(self, pos, move):
        return pos[0] + move.value[0], pos[1] + move.value[1]

    def _neighbors(self, pos, grid, stay=False):
        out = []
        for move in (MOVES_STAY if stay else MOVES):
            nxt = self._move(pos, move)
            if move == Move.STAY or self._open(nxt, grid):
                out.append((nxt, move))
        return out

    def _passable_neighbors(self, pos):
        out = []
        for move in MOVES:
            nxt = self._move(pos, move)
            if self._passable(nxt):
                out.append(nxt)
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
        dist = self._distance_map(start, grid, max_depth=max_depth)
        if goal in dist:
            return dist[goal]
        return max_depth + self._manhattan(start, goal)

    def _unknown_adjacent(self, pos):
        total = 0
        for move in MOVES:
            nxt = self._move(pos, move)
            if self._inside(nxt, self.known) and self.known[nxt] == -1:
                total += 1
        return total

    def _compute_articulation(self):
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


class PacmanAgent(BasePacmanAgent, SharedTools):
    """Identical algorithm to fixed_voronoi_agent's Pacman (Frontier Hunter)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Fusion Hunter"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self._init_state()

    def step(self, map_state, my_position, enemy_position, step_number):
        self._observe(map_state, my_position, enemy_position, DEFAULT_GHOST_START)

        if enemy_position is not None:
            path = self._bfs_path(my_position, enemy_position, self.known)
            if path:
                return self._path_action(my_position, path, map_state)

        target = None
        if self.fixed_map_mode and self.last_enemy is not None and self.last_enemy != my_position:
            target = self.last_enemy
        if target is None:
            target = self._frontier_target(my_position)
        if target is None and self.last_enemy is not None and self.last_enemy != my_position:
            target = self.last_enemy

        if target is not None:
            path = self._bfs_path(my_position, target, self.known)
            if path:
                return self._path_action(my_position, path, map_state)
        return self._fallback(my_position, map_state)

    def _frontier_target(self, start):
        distances = self._distance_map(start, self.known, max_depth=45)
        best = None
        best_score = -10**9
        for pos, dist in distances.items():
            if pos == start:
                continue
            score = self._unknown_adjacent(pos) * 12
            score -= dist * 0.8
            score -= self.visits.get(pos, 0) * 4
            if self.last_enemy is not None:
                score -= self._manhattan(pos, self.last_enemy) * 0.12
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
            score = self._unknown_adjacent(nxt) * 5 - self.visits.get(nxt, 0) * 2
            if score > best_score:
                best_score = score
                best_move = move
        return best_move, 1


class GhostAgent(BaseGhostAgent, SharedTools):
    """Cycle-aware Ghost. Same algorithm as cycle_agent."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.name = "Fusion Dancer"
        self.pacman_speed = max(1, int(kwargs.get("pacman_speed", 2)))
        self._init_state()
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
        self._compute_articulation()
        threat = enemy_position or self.last_enemy

        if self.fixed_map_mode and step_number <= len(self.fixed_opening):
            opening_move = self.fixed_opening[step_number - 1]
            opening_pos = self._move(my_position, opening_move)
            if self._open(opening_pos, map_state):
                return opening_move

        if threat is None:
            return self._wander(my_position, map_state)

        best_move = Move.STAY
        best_score = -10**18
        for nxt, move in self._neighbors(my_position, map_state, stay=True):
            score = self._evaluate(nxt, move, threat)
            if score > best_score:
                best_score = score
                best_move = move
        return best_move

    def _pacman_reach(self, pac_pos):
        out = [pac_pos]
        for move in MOVES:
            cur = pac_pos
            for _ in range(self.pacman_speed):
                nxt = self._move(cur, move)
                if not self._open(nxt, self.known):
                    break
                out.append(nxt)
                cur = nxt
        return out

    def _evaluate(self, pos, move, threat):
        pac_reach = self._pacman_reach(threat)
        min_direct = min(self._manhattan(pos, pp) for pp in pac_reach)
        if min_direct < 2:
            return -10**12

        exits = len(self._neighbors(pos, self.known))
        cycle = self._biconn_size.get(pos, 0) if self._biconn_size else 0
        is_art = self._art_points and pos in self._art_points

        score = 0.0
        score += min_direct * 9.0
        score += min(self._maze_distance(pos, threat, self.known, max_depth=35), 35) * 4.5
        score += exits * 8.0
        score += cycle * 1.2
        score += self._unknown_adjacent(pos) * 1.5
        score -= self.visits.get(pos, 0) * 2.5

        if is_art:
            score -= 40
        if exits <= 1:
            score -= 120
        elif exits == 2 and self._corridor_without_junction(pos):
            score -= 30
        if pos in list(self.recent)[-4:]:
            score -= 8
        if move == Move.STAY:
            score -= 14
        return score

    def _wander(self, pos, grid):
        best_score = -10**9
        best_move = Move.STAY
        for nxt, move in self._neighbors(pos, grid):
            cycle = self._biconn_size.get(nxt, 0) if self._biconn_size else 0
            score = cycle * 2 + len(self._neighbors(nxt, self.known)) * 6
            score += self._unknown_adjacent(nxt) * 3
            score -= self.visits.get(nxt, 0) * 2
            if score > best_score:
                best_score = score
                best_move = move
        return best_move
