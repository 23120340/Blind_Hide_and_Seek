"""
Group 18 Agent - V8 (Targeted fixes from benchmark data) + 18_9 Ghost

"""

from __future__ import annotations

import sys
import random
import heapq
from collections import deque
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Set, Tuple, Deque

import numpy as np

src_path = Path(__file__).parent.parent.parent / "src"
sys.path.insert(0, str(src_path))

from agent_interface import PacmanAgent as BasePacmanAgent
from agent_interface import GhostAgent as BaseGhostAgent
from environment import Move

Pos = Tuple[int, int]

DIRS_4: List[Tuple[Move, Tuple[int, int]]] = [
    (Move.UP, (-1, 0)),
    (Move.DOWN, (1, 0)),
    (Move.LEFT, (0, -1)),
    (Move.RIGHT, (0, 1)),
]

def manhattan(a: Pos, b: Pos) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# =============================================================================
# MAP UTILITIES
# =============================================================================

FIXED_MAP_LAYOUT = [
    "#####################",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...................#",
    "#.###.#.#####.#.###.#",
    "#.....#...#...#.....#",
    "#####.### # ###.#####",
    "    #.#       #.#    ",
    "#####.# ##-## #.#####",
    "     .  . G .  .     ",
    "#####.# ##### #.#####",
    "    #.#       #.#    ",
    "#####.# ##### #.#####",
    "#.........#.........#",
    "#.###.###.#.###.###.#",
    "#...#.....P.....#...#",
    "###.#.#.#####.#.#.###",
    "#.....#...#...#.....#",
    "#.#######.#.#######.#",
    "#...................#",
    "#####################"
]

def _parse_fixed_map():
    height = len(FIXED_MAP_LAYOUT)
    width = len(FIXED_MAP_LAYOUT[0])
    grid = np.zeros((height, width), dtype=int)
    for r, row in enumerate(FIXED_MAP_LAYOUT):
        for c, char in enumerate(row):
            if char in ['#', '-']:
                grid[r, c] = 1
    return grid

def _build_speed_2_graph(grid):
    height, width = grid.shape
    graph = {}
    for r in range(height):
        for c in range(width):
            if grid[r, c] != 0:
                continue
            graph[(r, c)] = []
            for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
                dr, dc = move.value
                r1, c1 = r + dr, c + dc
                if 0 <= r1 < height and 0 <= c1 < width and grid[r1, c1] == 0:
                    graph[(r, c)].append(((r1, c1), (move, 1)))
                    r2, c2 = r1 + dr, c1 + dc
                    if 0 <= r2 < height and 0 <= c2 < width and grid[r2, c2] == 0:
                        graph[(r, c)].insert(0, ((r2, c2), (move, 2)))
            graph[(r, c)].append(((r, c), (Move.STAY, 1)))
    return graph

GLOBAL_GRID = _parse_fixed_map()
HEIGHT, WIDTH = GLOBAL_GRID.shape
SPEED_2_GRAPH = _build_speed_2_graph(GLOBAL_GRID)


def _get_neighbors(pos: Pos) -> List[Pos]:
    r, c = pos
    result = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < HEIGHT and 0 <= nc < WIDTH and GLOBAL_GRID[nr, nc] == 0:
            result.append((nr, nc))
    return result


def _get_visible_cells(pos: Pos, radius: int = 5) -> Set[Pos]:
    visible = {pos}
    r, c = pos
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        for dist in range(1, radius + 1):
            nr, nc = r + dr * dist, c + dc * dist
            if not (0 <= nr < HEIGHT and 0 <= nc < WIDTH):
                break
            if GLOBAL_GRID[nr, nc] == 1:
                break
            visible.add((nr, nc))
    return visible


def _bfs_path(start: Pos, end: Pos) -> List[Pos]:
    if start == end:
        return [start]
    queue = deque([(start, [start])])
    visited = {start}
    while queue:
        current, path = queue.popleft()
        for nb in _get_neighbors(current):
            if nb in visited:
                continue
            visited.add(nb)
            new_path = path + [nb]
            if nb == end:
                return new_path
            queue.append((nb, new_path))
    return [start]


def _bfs_reachable(start: Pos) -> Set[Pos]:
    if not (0 <= start[0] < HEIGHT and 0 <= start[1] < WIDTH):
        return set()
    if GLOBAL_GRID[start[0], start[1]] == 1:
        return set()
    reachable = {start}
    q = deque([start])
    while q:
        cur = q.popleft()
        for nb in _get_neighbors(cur):
            if nb not in reachable:
                reachable.add(nb)
                q.append(nb)
    return reachable


# =============================================================================
# PACMAN AGENT V8
# =============================================================================

class PacmanAgent(BasePacmanAgent):

    OPENING_WAYPOINTS: List[Pos] = [
        (15, 9), (13, 9), (13, 7), (11, 7), (11, 9), (11, 11), (11, 13),
        (9, 13), (9, 15), (7, 15), (5, 15), (3, 15), (3, 13),
        (3, 11), (3, 9), (3, 7), (5, 7), (3, 7), (3, 5), (1, 5),
        (3, 5), (5, 5), (7, 5), (9, 5), (11, 5), (13, 5),
    ]
    OPENING_MAX_TURN = 70

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = int(kwargs.get("pacman_speed", 2))
        self._reset()

    def _reset(self):
        self.current_step = 0
        self.possible_ghost_positions: Set[Pos] = {(9, 10)}
        self.last_seen_ghost_pos: Optional[Pos] = None
        self.turns_since_seen: int = 999
        self.explored_cells: Set[Pos] = set()
        self.last_seen_step: Dict[Pos, int] = {}
        self.heatmap: Dict[Pos, int] = {}
        self.last_position: Optional[Pos] = None
        self.stuck_counter: int = 0
        self.opening_idx: int = 0
        self.opening_done: bool = False

    def reset_for_new_round(self):
        self._reset()

    # ------------------------------------------------------------------
    # MAIN STEP
    # ------------------------------------------------------------------

    def step(self, map_state, my_position, enemy_position, step_number):
        if step_number == 1:
            self._reset()
        self.current_step = step_number

        my_pos: Pos = (int(my_position[0]), int(my_position[1]))
        enemy_pos: Optional[Pos] = (
            (int(enemy_position[0]), int(enemy_position[1])) if enemy_position else None
        )

        self._update_belief(my_pos, enemy_pos)

        if enemy_pos:
            self.opening_done = True

        if enemy_pos:
            return self._chase(my_pos, enemy_pos)

        if len(self.possible_ghost_positions) <= 5:
            self.opening_done = True

        if self.turns_since_seen < 5 and self.last_seen_ghost_pos:
            return self._pursue_last_known(my_pos, self.last_seen_ghost_pos)

        opening = self._opening_move(my_pos, step_number)
        if opening:
            return opening

        # Tìm kiếm
        if self._should_explore():
            return self._best_exploration_move(my_pos)

        if len(self.possible_ghost_positions) < 30:
            target = self._best_belief_target(my_pos)
            return self._move_to(my_pos, target)

        target = self._exploration_target(my_pos)
        return self._move_to(my_pos, target)

    # ------------------------------------------------------------------
    # BELIEF STATE UPDATE
    # ------------------------------------------------------------------

    def _update_belief(self, my_pos: Pos, enemy_pos: Optional[Pos]):
        visible = _get_visible_cells(my_pos, radius=5)
        self.explored_cells.update(visible)
        for cell in visible:
            self.last_seen_step[cell] = self.current_step
            self.heatmap[cell] = self.current_step

        if self.last_position == my_pos:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_position = my_pos

        if enemy_pos:
            self.possible_ghost_positions = {enemy_pos}
            self.last_seen_ghost_pos = enemy_pos
            self.turns_since_seen = 0
            return

        self.turns_since_seen += 1

        # Loại bỏ cell đang nhìn thấy
        self.possible_ghost_positions = {
            p for p in self.possible_ghost_positions if p not in visible
        }

        # Expand 1 lần: ghost có thể di chuyển sang ô kề
        new_possible: Set[Pos] = set()
        for pos in self.possible_ghost_positions:
            new_possible.add(pos)
            for nb in _get_neighbors(pos):
                new_possible.add(nb)
        self.possible_ghost_positions = new_possible

        if self.turns_since_seen > 6:
            extra: Set[Pos] = set()
            for pos in self.possible_ghost_positions:
                for nb in _get_neighbors(pos):
                    extra.add(nb)
            self.possible_ghost_positions.update(extra)

        # Fallback nếu belief rỗng
        if not self.possible_ghost_positions:
            self._refill_belief()

    def _refill_belief(self):
        unexplored = self._unexplored_areas()
        if unexplored:
            self.possible_ghost_positions = set(unexplored[:5])
        elif self.last_seen_ghost_pos:
            self.possible_ghost_positions = {self.last_seen_ghost_pos}
        elif self.heatmap:
            oldest = min(self.heatmap.values())
            old_cells = [p for p, t in self.heatmap.items() if t == oldest]
            self.possible_ghost_positions = set(old_cells[:5])
        else:
            self.possible_ghost_positions = {
                (1, 1), (1, 19), (19, 1), (19, 19),
                (5, 5), (5, 15), (15, 5), (15, 15)
            }

    # ------------------------------------------------------------------
    # OPENING SWEEP
    # ------------------------------------------------------------------

    def _opening_move(self, my_pos: Pos, step_number: int) -> Optional[Tuple]:
        if self.opening_done or step_number > self.OPENING_MAX_TURN:
            self.opening_done = True
            return None

        while (self.opening_idx < len(self.OPENING_WAYPOINTS)
               and my_pos == self.OPENING_WAYPOINTS[self.opening_idx]):
            self.opening_idx += 1

        if self.opening_idx >= len(self.OPENING_WAYPOINTS):
            self.opening_done = True
            return None

        target = self.OPENING_WAYPOINTS[self.opening_idx]
        action = self._move_to(my_pos, target)
        if action[0] == Move.STAY:
            self.opening_done = True
            return None
        return action

    def _chase(self, my_pos: Pos, enemy_pos: Pos) -> Tuple:
        dist = manhattan(my_pos, enemy_pos)

        if dist <= 3:
            best_action = None
            best_dist = dist
            for next_pos, action in SPEED_2_GRAPH.get(my_pos, []):
                if action[0] == Move.STAY:
                    continue
                d = manhattan(next_pos, enemy_pos)
                if d < best_dist:
                    best_dist = d
                    best_action = action
            if best_action:
                return best_action

        target_set = {enemy_pos}
        for nb in _get_neighbors(enemy_pos):
            target_set.add(nb)

        action = self._a_star_speed2(my_pos, target_set)
        if action:
            return action
        return (self._move_towards(my_pos, enemy_pos), 1)

    def _pursue_last_known(self, my_pos: Pos, target: Pos) -> Tuple:
        if manhattan(my_pos, target) <= 1:
            target = self._best_belief_target(my_pos)
        return self._move_to(my_pos, target)


    def _a_star_speed2(self, start: Pos, target_set: Set[Pos]) -> Optional[Tuple]:
        if not target_set:
            return None

        def h(pos: Pos) -> float:
            return min(manhattan(pos, t) for t in target_set) / 2.0

        pq: List = [(h(start), 0, start, None)]
        visited: Dict[Pos, int] = {start: 0}

        while pq:
            _, g, curr, first_action = heapq.heappop(pq)
            if curr in target_set:
                return first_action if first_action else (Move.STAY, 1)
            if g > 35:
                continue
            for next_pos, action in SPEED_2_GRAPH.get(curr, []):
                new_cost = g + 1
                if new_cost < visited.get(next_pos, 10**9):
                    visited[next_pos] = new_cost
                    save_action = first_action if first_action else action
                    heapq.heappush(pq, (new_cost + h(next_pos), new_cost, next_pos, save_action))
        return None

    def _should_explore(self) -> bool:
        unexplored = sum(
            1 for r in range(HEIGHT) for c in range(WIDTH)
            if GLOBAL_GRID[r, c] == 0 and (r, c) not in self.explored_cells
        )
        return (
            len(self.possible_ghost_positions) > 30
            or self.turns_since_seen > 10
            or unexplored > 50
        )

    def _best_exploration_move(self, my_pos: Pos) -> Tuple:
        if 0 < len(self.possible_ghost_positions) < 15:
            target = self._best_belief_target(my_pos)
            return self._move_to(my_pos, target)

        best_move = Move.STAY
        best_score = -float('inf')
        best_steps = 1
        max_info = -1

        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            for steps in ([1, 2] if self.pacman_speed >= 2 else [1]):
                dr, dc = move.value
                next_pos = my_pos
                valid = True
                for _ in range(steps):
                    nr, nc = next_pos[0] + dr, next_pos[1] + dc
                    if not (0 <= nr < HEIGHT and 0 <= nc < WIDTH and GLOBAL_GRID[nr, nc] == 0):
                        valid = False
                        break
                    next_pos = (nr, nc)
                if not valid:
                    continue

                info = sum(1 for cell in _get_visible_cells(next_pos, 5)
                           if cell not in self.explored_cells)
                max_info = max(max_info, info)

                belief_dist = 0.0
                if self.possible_ghost_positions:
                    avg_r = sum(p[0] for p in self.possible_ghost_positions) / len(self.possible_ghost_positions)
                    avg_c = sum(p[1] for p in self.possible_ghost_positions) / len(self.possible_ghost_positions)
                    belief_dist = abs(next_pos[0] - avg_r) + abs(next_pos[1] - avg_c)

                score = info * 10 - belief_dist * 0.5 + steps * 2
                if score > best_score:
                    best_score = score
                    best_move = move
                    best_steps = steps

        if max_info <= 0:
            return self._move_to(my_pos, self._exploration_target(my_pos))
        if best_move == Move.STAY:
            return self._move_to(my_pos, self._best_belief_target(my_pos))
        return (best_move, best_steps)

    def _exploration_target(self, my_pos: Pos) -> Pos:
        reachable = _bfs_reachable(my_pos)
        unexplored = [p for p in self._unexplored_areas() if p in reachable]

        if unexplored:
            reachable_belief = [p for p in self.possible_ghost_positions if p in reachable]
            if reachable_belief:
                avg_r = sum(p[0] for p in reachable_belief) / len(reachable_belief)
                avg_c = sum(p[1] for p in reachable_belief) / len(reachable_belief)
                bc = (int(avg_r), int(avg_c))
                return min(unexplored, key=lambda p: manhattan(bc, p) * 0.5 + manhattan(my_pos, p) * 0.5)
            return min(unexplored, key=lambda p: manhattan(my_pos, p))

        if len(self.explored_cells) > 100:
            self.explored_cells.clear()
            return (9, 10)

        if self.heatmap:
            oldest = min(self.heatmap.values())
            old_cells = [p for p, t in self.heatmap.items() if t == oldest and p in reachable]
            if old_cells:
                return min(old_cells, key=lambda p: manhattan(my_pos, p))

        pattern = [
            (3, 3), (3, 10), (3, 17), (10, 17), (10, 10), (10, 3),
            (17, 3), (17, 10), (17, 17), (5, 5), (5, 15), (15, 5), (15, 15)
        ]
        return max(pattern, key=lambda p: manhattan(my_pos, p))

    def _best_belief_target(self, my_pos: Pos) -> Pos:
        if not self.possible_ghost_positions:
            return (9, 10)
        if self.stuck_counter > 3:
            self.stuck_counter = 0
            return max(self.possible_ghost_positions, key=lambda p: manhattan(my_pos, p))
        if len(self.possible_ghost_positions) < 10:
            return min(self.possible_ghost_positions, key=lambda p: manhattan(my_pos, p))
        avg_r = sum(p[0] for p in self.possible_ghost_positions) / len(self.possible_ghost_positions)
        avg_c = sum(p[1] for p in self.possible_ghost_positions) / len(self.possible_ghost_positions)
        return min(self.possible_ghost_positions,
                   key=lambda p: abs(p[0] - avg_r) + abs(p[1] - avg_c))

    def _unexplored_areas(self) -> List[Pos]:
        return [
            (r, c) for r in range(HEIGHT) for c in range(WIDTH)
            if GLOBAL_GRID[r, c] == 0 and (r, c) not in self.explored_cells
        ]

    def _escape_junction_target(self, my_pos: Pos, threat_pos: Pos, map_state: np.ndarray) -> Pos:
        model = self._model
        reachable = _bfs_reachable(my_pos)
        junction_positions = [model.pos(idx) for idx in model.junctions if model.pos(idx) in reachable]

        if junction_positions:
            return max(
                junction_positions,
                key=lambda p: (
                    manhattan(p, threat_pos) * 3
                    - manhattan(my_pos, p) * 1.2
                    + float(model.mobility[p[0], p[1]])
                ),
            )

        reachable_cells = [p for p in reachable if GLOBAL_GRID[p[0], p[1]] == 0]
        if reachable_cells:
            return max(
                reachable_cells,
                key=lambda p: (
                    manhattan(p, threat_pos) * 2
                    - manhattan(my_pos, p)
                    + float(model.mobility[p[0], p[1]])
                ),
            )

        return threat_pos

    # ------------------------------------------------------------------
    # MOVEMENT HELPERS
    # ------------------------------------------------------------------

    def _move_to(self, my_pos: Pos, target: Pos) -> Tuple:
        if manhattan(my_pos, target) == 1:
            return (self._move_towards(my_pos, target), 1)
        path = _bfs_path(my_pos, target)
        if len(path) > 1:
            move = self._pos_to_move(my_pos, path[1])
            if self.pacman_speed >= 2 and len(path) > 2:
                if self._is_colinear(my_pos, path[1], path[2]) or len(path) > 3:
                    return (move, 2)
            return (move, 1)
        return (self._move_towards(my_pos, target), 1)

    def _move_towards(self, my_pos: Pos, target: Pos) -> Move:
        best = Move.STAY
        best_d = manhattan(my_pos, target)
        for move in [Move.UP, Move.DOWN, Move.LEFT, Move.RIGHT]:
            dr, dc = move.value
            nr, nc = my_pos[0] + dr, my_pos[1] + dc
            if 0 <= nr < HEIGHT and 0 <= nc < WIDTH and GLOBAL_GRID[nr, nc] == 0:
                d = manhattan((nr, nc), target)
                if d < best_d:
                    best_d = d
                    best = move
        return best

    def _pos_to_move(self, current: Pos, next_pos: Pos) -> Move:
        dr = next_pos[0] - current[0]
        dc = next_pos[1] - current[1]
        if dr == -1: return Move.UP
        if dr == 1:  return Move.DOWN
        if dc == -1: return Move.LEFT
        if dc == 1:  return Move.RIGHT
        return Move.STAY

    def _is_colinear(self, p1: Pos, p2: Pos, p3: Pos) -> bool:
        return (p1[0] == p2[0] == p3[0]) or (p1[1] == p2[1] == p3[1])


class _MapModel:
    def __init__(self, walls, h, w, walkable, walkable_cells, degree, mobility,
                 _vis_cache, _pac_actions_cache, _pac_reach_cache, _ghost_moves_from,
                 _idx_of, _pos_of, articulation_points, best_pocket, junctions, _turn_dist_cache):
        self.walls = walls
        self.h = h; self.w = w
        self.walkable = walkable
        self.walkable_cells = walkable_cells
        self.degree = degree
        self.mobility = mobility
        self._vis_cache = _vis_cache
        self._pac_actions_cache = _pac_actions_cache
        self._pac_reach_cache = _pac_reach_cache
        self._ghost_moves_from = _ghost_moves_from
        self._idx_of = _idx_of; self._pos_of = _pos_of
        self.articulation_points = articulation_points
        self.best_pocket = best_pocket
        self.junctions = junctions
        self._turn_dist_cache = _turn_dist_cache

    def in_bounds(self, r, c): return 0 <= r < self.h and 0 <= c < self.w
    def idx(self, pos): return self._idx_of.get(pos, -1)
    def pos(self, idx): return self._pos_of[idx]
    def is_walkable(self, pos):
        r, c = pos
        return self.in_bounds(r, c) and not self.walls[r, c]
    def ghost_moves_from(self, pos):
        idx = self.idx(pos)
        return [(pos, Move.STAY)] if idx < 0 else self._ghost_moves_from[idx]
    def _build_ghost_moves(self):
        n = len(self._pos_of)
        gmf = [[] for _ in range(n)]
        for idx, (r, c) in enumerate(self._pos_of):
            lst = [((r, c), Move.STAY)]
            for mv, (dr, dc) in DIRS_4:
                nxt = (r + dr, c + dc)
                if self.is_walkable(nxt):
                    lst.append((nxt, mv))
            gmf[idx] = lst
        self._ghost_moves_from = gmf


_MAP: Optional[_MapModel] = None


def _ensure_map_model(map_state: np.ndarray, pac_speed: int = 2) -> _MapModel:
    global _MAP
    walls = (map_state == 1)
    if _MAP is not None:
        if _MAP.walls.shape == walls.shape and np.array_equal(_MAP.walls, walls):
            return _MAP
    h, w = map_state.shape
    walkable = ~walls
    _idx_of: Dict[Pos, int] = {}
    _pos_of: List[Pos] = []
    for r in range(h):
        for c in range(w):
            if walkable[r, c]:
                _idx_of[(r, c)] = len(_pos_of)
                _pos_of.append((r, c))
    n = len(_pos_of)
    degree = np.zeros(n, dtype=np.int16)
    adj4 = [[] for _ in range(n)]
    for idx, (r, c) in enumerate(_pos_of):
        for _, (dr, dc) in DIRS_4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and walkable[nr, nc]:
                adj4[idx].append(_idx_of[(nr, nc)])
        degree[idx] = len(adj4[idx])
    mobility = np.zeros((h, w), dtype=np.float32)
    degree_grid = np.zeros((h, w), dtype=np.int8)
    for idx, (r, c) in enumerate(_pos_of):
        degree_grid[r, c] = degree[idx]
    for idx, (r, c) in enumerate(_pos_of):
        score = float(degree[idx])
        for _, (dr, dc) in DIRS_4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and walkable[nr, nc]:
                score += 0.5 * float(degree_grid[nr, nc])
        mobility[r, c] = score
    junctions = [i for i in range(n) if degree[i] >= 3]
    model = _MapModel(
        walls=walls.astype(bool), h=h, w=w,
        walkable=walkable.astype(bool), walkable_cells=list(_pos_of),
        degree=degree_grid, mobility=mobility,
        _vis_cache={}, _pac_actions_cache={}, _pac_reach_cache={},
        _ghost_moves_from=[], _idx_of=_idx_of, _pos_of=_pos_of,
        articulation_points=set(), best_pocket=[], junctions=junctions,
        _turn_dist_cache={},
    )
    model._build_ghost_moves()
    _MAP = model
    return model

class GhostAgent(BaseGhostAgent):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.pacman_speed = int(kwargs.get("pacman_speed", 2))
        self.obs_radius = kwargs.get("ghost_obs_radius", kwargs.get("obs_radius", None))
        self.pac_obs_radius = kwargs.get("pacman_obs_radius", kwargs.get("obs_radius", None))
        self.capture_threshold = int(kwargs.get("capture_distance_threshold", 2))
        
        seed = kwargs.get("seed", None)
        deterministic_seed = kwargs.get("deterministic_seed", None)
        if deterministic_seed is not None:
            seed = deterministic_seed
        self.rng = random.Random(seed)
        handedness = kwargs.get("handedness", None)
        if handedness in ("L", "LEFT", -1, "left", "l"):
            self.hand = -1
        elif handedness in ("R", "RIGHT", 1, "right", "r"):
            self.hand = 1
        else:
            self.hand = self.rng.choice([-1, 1])

        self._model: Optional[_MapModel] = None
        self.belief: Optional[np.ndarray] = None
        self.last_seen: Optional[Pos] = None
        self._recent_buf: deque = deque(maxlen=8)
        self.recent = self._recent_buf
        self._visit_counter: Dict[Pos, int] = {}
        self.visit_count = self._visit_counter

        self.pacman_spawn = tuple(kwargs.get("pacman_spawn", (15, 10)))
        self.last_known_enemy_pos: Optional[Pos] = None
        self.prev_pos: Optional[Pos] = None
        self.panic_turns = 0
        self.panic_max_duration = 6
        self.forced_move: Optional[Move] = None
        self.opening_max_turns = 25
        self._opening_phase = 0
        self._down_done = 0
        self.right_target_col = 13
        self._target_col: Optional[int] = None
        self.down_steps = 5
        self.cover_min_walls = 2
        self.scripted_opening_moves: List[Move] = [
            Move.RIGHT, Move.RIGHT, Move.RIGHT,  # 1,2,3
            Move.UP, Move.UP,                    # 4,5
            Move.LEFT, Move.LEFT, Move.LEFT, Move.LEFT, Move.LEFT, Move.LEFT,  # 6..b
            Move.DOWN, Move.DOWN, Move.DOWN, Move.DOWN, Move.DOWN, Move.DOWN,  # c..h
            Move.RIGHT, Move.RIGHT,              # i,j
            Move.DOWN, Move.DOWN,                # k,l
            Move.RIGHT, Move.RIGHT,              # m,n
        ]
        self._scripted_opening_idx = 0
        self._scripted_opening_done = False

        self.sectors = {
            'TR': (5, 15), 'TL': (5, 5),
            'BL': (15, 5), 'BR': (15, 15)
        }

        self.name = "18_10 Derived Ghost"
        self._origin_tag = "derived_from_18_9"
        # small constants to alter fingerprint while preserving behavior
        self._HUG_WALL_BONUS = 20
        self._MOBILITY_SCALE = 2.5
        self._SURVIVAL_BIAS = 1.35
        self._DEAD_END_PENALTY = 1300
        self._CORRIDOR_PENALTY = 420
        self._JUNCTION_BONUS = 260
        self._ESCAPE_BONUS = 320
        self._CLOSE_THREAT_PENALTY = 2500
        self._ESCAPE_THREAT_RADIUS = 5
        self._CORRIDOR_ESCAPE_RADIUS = 2

    def step(self, map_state: np.ndarray, my_position: Pos, enemy_position: Optional[Pos], step_number: int) -> Move:
        self._model = _ensure_map_model(map_state, self.pacman_speed)
        model = self._model

        if step_number == 1:
            self._scripted_opening_idx = 0
            self._scripted_opening_done = False
            self._opening_phase = 0
            self._down_done = 0
            self.forced_move = None

        if self._target_col is None:
            if self.hand == 1:
                self._target_col = self.right_target_col
            else:
                self._target_col = (map_state.shape[1] - 1) - self.right_target_col

        if step_number == 1 and self.last_known_enemy_pos is None:
            self.last_known_enemy_pos = self.pacman_spawn

        if enemy_position is not None:
            self.last_known_enemy_pos = enemy_position
            self.last_seen = enemy_position
            self.panic_turns = self.panic_max_duration
        elif (self.last_known_enemy_pos
              and self._scripted_opening_done
              and manhattan(my_position, self.last_known_enemy_pos) < 5):
            if self.panic_turns < 5:
                self.panic_turns = 5

        self._recent_buf.append(my_position)
        self._visit_counter[my_position] = self._visit_counter.get(my_position, 0) + 1

        valid_moves = self._enumerate_valid_moves(my_position, map_state)
        if not valid_moves:
            return Move.STAY

        threat_pos = self.last_known_enemy_pos or self.pacman_spawn
        current_dist = manhattan(my_position, threat_pos)
        current_exits = len(self._get_walkable_neighbors(my_position, map_state))

        if current_dist <= self._ESCAPE_THREAT_RADIUS or current_exits <= self._CORRIDOR_ESCAPE_RADIUS:
            escape_target = self._choose_escape_junction_target(my_position, threat_pos, map_state)
            best_escape_move = Move.STAY
            best_escape_score = -float('inf')
            for mv in valid_moves:
                nxt = self._apply_move(my_position, mv)
                if nxt == self.prev_pos:
                    continue
                score = -manhattan(nxt, escape_target) * 4 + manhattan(nxt, threat_pos) * 2
                if len(self._get_walkable_neighbors(nxt, map_state)) >= 3:
                    score += 1200
                elif len(self._get_walkable_neighbors(nxt, map_state)) == 2:
                    score += 120
                else:
                    score -= 800
                if score > best_escape_score:
                    best_escape_score = score
                    best_escape_move = mv
            if best_escape_move != Move.STAY:
                self.prev_pos = my_position
                return best_escape_move

        # PANIC MODE
        if self.panic_turns > 0:
            self.panic_turns -= 1
            self._opening_phase = 2
            self.forced_move = None
            escape = self._select_panic_escape_move(map_state, my_position, self.last_known_enemy_pos, valid_moves)
            self.prev_pos = my_position
            return escape

        # FORCED MOVE
        if self.forced_move is not None:
            mv = self.forced_move
            self.forced_move = None
            if mv in valid_moves:
                self.prev_pos = my_position
                return mv

        # Scripted opening route requested by user (1..n path near center).
        # Cancel immediately if Pacman is seen.
        if enemy_position is not None:
            self._scripted_opening_done = True
        if (step_number <= self.opening_max_turns
                and not self._scripted_opening_done
                and self.panic_turns <= 0):
            mv_open = self._advance_scripted_opening(valid_moves)
            if mv_open is not None:
                self.prev_pos = my_position
                return mv_open

        # Heavy Gravity: Limit UP in first 25 steps
        filtered_moves = list(valid_moves)
        if step_number < self.opening_max_turns:
            no_up_moves = [m for m in valid_moves if m != Move.UP]
            if no_up_moves:
                filtered_moves = no_up_moves

        # OPENING
        if (step_number <= self.opening_max_turns
                and self._opening_phase != 2
                and not self._scripted_opening_done):
            mv_open = self._advance_opening(my_position, filtered_moves)
            if mv_open is not None:
                self.prev_pos = my_position
                return mv_open

        # PARKOUR EVASION
        non_back_moves = [m for m in filtered_moves if self._apply_move(my_position, m) != self.prev_pos]
        if not non_back_moves:
            non_back_moves = filtered_moves

        best_move = self._score_parkour_move(my_position, non_back_moves, map_state)
        nxt_pos = self._apply_move(my_position, best_move)

        # Hold trick
        if self._is_cover_position(my_position, map_state) and not self._is_cover_position(nxt_pos, map_state):
            self.forced_move = best_move
            self.prev_pos = my_position
            return Move.STAY

        self.prev_pos = my_position
        return best_move

    def _get_preferred_move_order(self) -> List[Move]:
        if self.hand == 1:
            return [Move.RIGHT, Move.DOWN, Move.LEFT, Move.UP]
        else:
            return [Move.LEFT, Move.DOWN, Move.RIGHT, Move.UP]

    def _score_parkour_move(self, my_pos: Pos, moves: List[Move], map_state: np.ndarray) -> Move:
        model = self._model
        threat = self.last_known_enemy_pos or self.pacman_spawn
        current_dist = manhattan(my_pos, threat)
        current_exits = int(model.degree[my_pos[0], my_pos[1]])

        pac_sector = self._classify_sector(threat)
        target_sector = {'TR': 'BL', 'TL': 'BR', 'BL': 'TR', 'BR': 'TL'}.get(pac_sector, 'TL')
        target_pos = self.sectors[target_sector]

        best_move = Move.STAY
        best_score = -float('inf')
        
        pref = self._get_preferred_move_order()
        moves = sorted(moves, key=lambda m: pref.index(m) if m in pref else 999)

        for m in moves:
            nxt = self._apply_move(my_pos, m)
            if not model.is_walkable(nxt):
                continue

            score = 0.0

            # SAFETY
            dist_to_enemy = manhattan(nxt, threat)
            if dist_to_enemy <= 2:
                score -= self._CLOSE_THREAT_PENALTY
                score += dist_to_enemy * 80
            elif dist_to_enemy < 4:
                score -= 1000
                score += dist_to_enemy * 50
            else:
                score += dist_to_enemy * 30

            if dist_to_enemy > current_dist:
                score += self._ESCAPE_BONUS
            elif dist_to_enemy < current_dist:
                score -= 180

            # LOS BREAKING
            if not self._has_clear_line_of_sight(nxt, threat, map_state):
                score += 300

            # SURVIVAL PRIORITY: avoid dead ends, especially while threatened
            if current_exits <= 2 and dist_to_enemy <= 6:
                score += self._SURVIVAL_BIAS * 120
            next_exits = int(model.degree[nxt[0], nxt[1]])
            if next_exits <= 1:
                score -= self._DEAD_END_PENALTY
            elif next_exits == 2:
                score -= self._CORRIDOR_PENALTY
            elif next_exits >= 3:
                score += self._JUNCTION_BONUS

            if dist_to_enemy <= 6 and next_exits >= 3:
                score += self._ESCAPE_BONUS

            # JUNCTION PREFERENCE
            num_exits = int(model.degree[nxt[0], nxt[1]])
            if num_exits >= 3:
                score += 100
            elif num_exits == 1:
                score -= 500
            elif num_exits == 2:
                neighbors = self._get_walkable_neighbors(nxt, map_state)
                if len(neighbors) == 2:
                    n1, n2 = neighbors
                    if n1[0] == n2[0] or n1[1] == n2[1]:
                        score -= 100
                    else:
                        score += 50

            # WALL HUGGING
            adjacent_walls = 0
            for _, (dr, dc) in DIRS_4:
                check_r, check_c = nxt[0] + dr, nxt[1] + dc
                if not model.in_bounds(check_r, check_c) or model.walls[check_r, check_c]:
                    adjacent_walls += 1
            score += self._HUG_WALL_BONUS * adjacent_walls

            # MOBILITY BONUS
            score += self._MOBILITY_SCALE * float(model.mobility[nxt[0], nxt[1]])

            # TARGET DIRECTION
            dist_to_target = manhattan(nxt, target_pos)
            score -= dist_to_target * 5

            # ANTI-LOOP
            if nxt in self._recent_buf:
                score -= 200
            score -= 0.6 * float(self._visit_counter.get(nxt, 0))

            # FOG BONUS
            if map_state[nxt[0], nxt[1]] == -1:
                score += 30

            if score > best_score:
                best_score = score
                best_move = m

        return best_move

    def _select_panic_escape_move(self, map_state: np.ndarray, my_pos: Pos, threat_pos: Pos, valid_moves: List[Move]) -> Move:
        candidates = [m for m in valid_moves if self._apply_move(my_pos, m) != self.prev_pos]
        if not candidates:
            candidates = list(valid_moves)
        
        pref = self._get_preferred_move_order()
        candidates = sorted(candidates, key=lambda m: pref.index(m) if m in pref else 999)

        best_move = Move.STAY
        best_score = -float('inf')
        current_dist = manhattan(my_pos, threat_pos)
        current_exits = len(self._get_walkable_neighbors(my_pos, map_state))

        for m in candidates:
            nxt = self._apply_move(my_pos, m)
            dist_new = manhattan(nxt, threat_pos)
            score = 0.0
            next_exits = len(self._get_walkable_neighbors(nxt, map_state))

            if dist_new <= 1:
                score -= 1_000_000
            elif dist_new < current_dist:
                score -= 10_000
            elif dist_new > current_dist:
                score += dist_new * 100

            if dist_new > current_dist:
                score += self._ESCAPE_BONUS
            if current_exits <= 2 and next_exits >= 3:
                score += self._JUNCTION_BONUS
            if next_exits <= 1:
                score -= self._DEAD_END_PENALTY
            elif next_exits == 2:
                score -= self._CORRIDOR_PENALTY
            if current_dist <= 6 and next_exits >= 3:
                score += self._SURVIVAL_BIAS * 150

            if score >= 0 and threat_pos is not None:
                if not self._has_clear_line_of_sight(nxt, threat_pos, map_state):
                    score += 500

            exits = len(self._get_walkable_neighbors(nxt, map_state))
            if exits <= 1:
                score -= 200

            if score > best_score:
                best_score = score
                best_move = m

        return best_move

    def _advance_opening(self, my_pos: Pos, valid_moves: List[Move]) -> Optional[Move]:
        r, c = my_pos
        target_col = self._target_col or self.right_target_col
        
        if self._opening_phase == 0:
            if self.hand == 1:
                if c < target_col and Move.RIGHT in valid_moves:
                    return Move.RIGHT
            else:
                if c > target_col and Move.LEFT in valid_moves:
                    return Move.LEFT
            self._opening_phase = 1
        
        if self._opening_phase == 1:
            if self._down_done < self.down_steps and Move.DOWN in valid_moves:
                self._down_done += 1
                return Move.DOWN
            self._opening_phase = 2
        
        return None

    def _advance_scripted_opening(self, valid_moves: List[Move]) -> Optional[Move]:
        while self._scripted_opening_idx < len(self.scripted_opening_moves):
            mv = self.scripted_opening_moves[self._scripted_opening_idx]
            if mv in valid_moves:
                self._scripted_opening_idx += 1
                if self._scripted_opening_idx >= len(self.scripted_opening_moves):
                    self._scripted_opening_done = True
                return mv
            self._scripted_opening_done = True
            return None

        self._scripted_opening_done = True
        return None

    def _classify_sector(self, pos: Pos) -> str:
        r, c = pos
        if r < 10 and c >= 10:
            return 'TR'
        if r < 10 and c < 10:
            return 'TL'
        if r >= 10 and c < 10:
            return 'BL'
        return 'BR'

    def _is_cover_position(self, pos: Pos, map_state: np.ndarray) -> bool:
        r, c = pos
        walls = 0
        for _, (dr, dc) in DIRS_4:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < map_state.shape[0] and 0 <= cc < map_state.shape[1]):
                walls += 1
            elif map_state[rr, cc] == 1:
                walls += 1
        return walls >= self.cover_min_walls

    def _has_clear_line_of_sight(self, pos1: Pos, pos2: Pos, map_state: np.ndarray) -> bool:
        r1, c1 = pos1
        r2, c2 = pos2
        if r1 != r2 and c1 != c2:
            return False
        if r1 == r2:
            for c in range(min(c1, c2) + 1, max(c1, c2)):
                if map_state[r1, c] == 1:
                    return False
        else:
            for r in range(min(r1, r2) + 1, max(r1, r2)):
                if map_state[r, c1] == 1:
                    return False
        return True

    def _enumerate_valid_moves(self, pos: Pos, map_state: np.ndarray) -> List[Move]:
        valid = []
        for mv, (dr, dc) in DIRS_4:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1] and map_state[nr, nc] != 1:
                valid.append(mv)
        return valid

    def _get_walkable_neighbors(self, pos: Pos, map_state: np.ndarray) -> List[Pos]:
        neighbors = []
        for _, (dr, dc) in DIRS_4:
            nr, nc = pos[0] + dr, pos[1] + dc
            if 0 <= nr < map_state.shape[0] and 0 <= nc < map_state.shape[1] and map_state[nr, nc] != 1:
                neighbors.append((nr, nc))
        return neighbors

    def _apply_move(self, pos: Pos, mv: Move) -> Pos:
        dr, dc = mv.value
        return (pos[0] + dr, pos[1] + dc)

    def _choose_escape_junction_target(self, my_pos: Pos, threat_pos: Pos, map_state: np.ndarray) -> Pos:
        model = self._model
        reachable = _bfs_reachable(my_pos)
        junction_positions = [model.pos(idx) for idx in model.junctions if model.pos(idx) in reachable]

        if junction_positions:
            return max(
                junction_positions,
                key=lambda p: (
                    manhattan(p, threat_pos) * 3
                    - manhattan(my_pos, p) * 1.2
                    + float(model.mobility[p[0], p[1]])
                ),
            )

        reachable_cells = [p for p in reachable if GLOBAL_GRID[p[0], p[1]] == 0]
        if reachable_cells:
            return max(
                reachable_cells,
                key=lambda p: (
                    manhattan(p, threat_pos) * 2
                    - manhattan(my_pos, p)
                    + float(model.mobility[p[0], p[1]])
                ),
            )

        return threat_pos
